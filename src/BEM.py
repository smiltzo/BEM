import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import BEM_utils as func
import induction_module as ind
from BEM_dataclasses import WTG, Wind, Simulation, AeroData
from scipy.interpolate import RegularGridInterpolator

# Dataclasses
wtg = WTG()
sim = Simulation()
wind = Wind()
aero = AeroData(wtg, sim, [f"FFA-W3-{n}.csv" for n in wtg.thicknesses])

wind.hasTowerEffect = True
wind.hasShear = True
wtg.yaw = np.deg2rad(20.0)
# ------------------------
# TRANSFORMATION MATRICES
# ------------------------

# From ground → nacelle (yaw and tilt)
a1 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(wtg.yaw), np.sin(wtg.yaw)],
    [0.0, -np.sin(wtg.yaw), np.cos(wtg.yaw)]
])

a2 = np.array([
    [np.cos(wtg.tilt), 0.0, -np.sin(wtg.tilt)],
    [0.0, 1.0, 0.0],
    [np.sin(wtg.tilt), 0.0, np.cos(wtg.tilt)]
])

a3 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, np.cos(wtg.roll), np.sin(wtg.roll)],
    [0.0, -np.sin(wtg.roll), np.cos(wtg.roll)]
])

a12 = a3 @ a2 @ a1   # ground → nacelle

# Shaft → blade (cone)
a34 = np.array([
    [np.cos(wtg.cone), 0.0, -np.sin(wtg.cone)],
    [0.0, 1.0, 0.0],
    [np.sin(wtg.cone), 0.0, np.cos(wtg.cone)]
])

debugIdx = (8, 0, 8)

# ------------------------
# TIME LOOP
# ------------------------
for i in range(sim.nSteps):
    
    # BLADE LOOP
    for j in range(wtg.blades):
        sim.theta[j, i + 1] = sim.theta[j, i] + sim.omega * sim.dt
        a23_blade1 = func.get_a23(sim.theta[j, i + 1])

        # BLADE ELEMENT LOOP
        for e in range(sim.bladeElements - 1):
            idx1 = (e, j, i + 1)
            idx  = (e, j, i)
            
            sim.position[:, *idx1] = func.get_position(wtg, sim, a12, a23_blade1, a34).squeeze()
            V_local = np.array([[0.0, 0.0, wind.V0_z]]).T

            if wind.hasShear:
                V_local[2, 0] = func.wind_shear(wind, wtg, sim.position[0, *idx1])

            if idx1 == debugIdx:
                hello = "Debug"

            if wind.hasTowerEffect:
                wtg.update_tower_radius(sim.position[0, *idx1])
                wind.Vz = V_local[2, 0]
                V_local, isStagnant = func.tower_model(wind, wtg,
                                      sim.position[1, *idx1],
                                      sim.position[2, *idx1])  
                if isStagnant:
                    V_local = sim.windSpeed[:, *idx]

            sim.windSpeed[:, *idx1] = func.go_to_ground_system(a12, a23_blade1, a34,
                                                                     V_local).squeeze()
            # sim.windSpeed[: ,*idx1] = V_local.squeeze()
            
            # BEM stuff - Induced wind
            aero.relWindSpeed[1, *idx1] = (sim.windSpeed[1, *idx1] + aero.inducedWind[1, *idx] - sim.omega * wtg.bladeData["r"][e] * wtg.cone).squeeze()
            aero.relWindSpeed[2, *idx1] = (sim.windSpeed[2, *idx1] + aero.inducedWind[2, *idx]).squeeze()

            # Nan Check
            # arr = aero.inducedWind
            # nanCheck = np.isnan(arr)
            # print("NaN in induced wind Z:", np.sum(nanCheck))
            # print("NaN by axis 0 (components):", np.sum(nanCheck, axis=(1,2,3)))
            # print(f"NaNs in blade elemtnt: {np.sum(nanCheck, axis=(0, 2, 3))}")
            # print(f"NaNs in blade: {np.sum(nanCheck, axis=(0, 1, 3))}")
            # print(f"Nans in time: {np.sum(nanCheck, axis=(0, 1, 2))}")

            # right after aero.relWindSpeed assignment
            vals = dict(
                windSpeed1 = sim.windSpeed[1, *idx1],
                windSpeed2 = sim.windSpeed[2, *idx1],
                induced1 = aero.inducedWind[1, *idx],
                induced2 = aero.inducedWind[2, *idx],
                rel1 = aero.relWindSpeed[1, *idx1],
                rel2 = aero.relWindSpeed[2, *idx1],
            )
            for k, v in vals.items():
                if np.isnan(v) or np.isinf(v):
                    print(f"NaN/inf detected at idx={idx1}: {k}={v}")
                    # optionally print more context:
                    print("sim.windSpeed slice:", sim.windSpeed[:, :, :, :].shape)  # or more targeted prints
                    raise RuntimeError("Debug: NaN/Inf detected")

            try:
                flowAngle = func.flow_angle(-aero.relWindSpeed[2, *idx1],
                                                            aero.relWindSpeed[1, *idx1]) # Nan here
            except ValueError as e:
                print(f"Error computing flow angle at idx {idx1}: {e}")
                print(f"Vrel_y: {aero.relWindSpeed[1, *idx1]}, Vrel_z: {aero.relWindSpeed[2, *idx1]}")
                flowAngle = 0.0

            aero.flowAngle[*idx1] = flowAngle

            # AoA in degrees
            aero.AoA[*idx1] = np.rad2deg(aero.flowAngle[*idx1]) - (wtg.bladeData["twist"][e] + wtg.pitch0)

            # Nan here
            relWindSpeedMagnitude = np.hypot(aero.relWindSpeed[1, *idx1], aero.relWindSpeed[2, *idx1])
            
            # Read and interpolate lift and drag
            point = np.atleast_2d((aero.AoA[*idx1], wtg.bladeData["thick"][e]))
            Cd = func.interpolate_drag(aero, wtg, point, idx1)

            if sim.dynamicStall:
                # S. Øye dyn. stall model
                cl_inv, f_s, cl_fs = func.interpolate_lift_dyn_stall(aero, wtg, point)

                tau = 4 * wtg.bladeData["chord"][e] / relWindSpeedMagnitude
                aero.separationFactor[*idx1] = f_s + (aero.separationFactor[*idx] - f_s) * np.exp(-sim.dt / tau)
                Cl = aero.separationFactor[*idx1] * cl_inv + (1 - aero.separationFactor[*idx1]) * cl_fs
            else:
                # Just steady-state Cl
                Cl = func.interpolate_lift(aero, wtg, point)

            # Compute element Lift and Drag
            lift = 0.5 * wind.density * relWindSpeedMagnitude**2 * Cl * wtg.bladeData["chord"][e]
            drag = 0.5 * wind.density * relWindSpeedMagnitude**2 * Cd * wtg.bladeData["chord"][e]
            
            aero.lift[*idx1] = lift.squeeze()
            aero.drag[*idx1] = drag.squeeze()

            # Rotate them into rotor plane (by the pitch angle)
            pNormal =  lift * np.cos(aero.flowAngle[*idx1]) + drag * np.sin(aero.flowAngle[*idx1])
            pTangent = drag * np.sin(aero.flowAngle[*idx1]) - lift * np.cos(aero.flowAngle[*idx1])
            
            aero.normalForce[*idx1] = pNormal.squeeze()
            aero.tangentialForce[*idx1] = pTangent.squeeze()

            # ---- DYNAMIC INFLOW ----
            # Compute Quasi-Steady induced winds
            # aero.inducedWindQS[1:, *idx1] = ind.compute_quasi_steady_induction(aero, wtg, wind, sim, idx1).squeeze()
            if idx1 == debugIdx:
                hello = "Debug"
            # ind.compute_induction(aero, sim, wtg, wind, idx1)
            end_inner_loop = "Done"



calculations = "Done"
# Forces, moments and power calculation
shape = (wtg.blades, sim.nSteps + 1)

F_normal = np.zeros(shape)
F_tangent = np.zeros(shape)
Thrust = np.zeros(shape)
Torque = np.zeros(shape)

dr = np.gradient(wtg.bladeData["r"])
for b in range(wtg.blades):
    F_normal[b, :] = np.sum(aero.normalForce[:, b, :] * dr[:, None], axis=0)
    F_tangent[b, :] = np.sum(aero.tangentialForce[:, b, :] * dr[:, None], axis=0)

    Thrust[b, :] = F_normal[b, :]
    Torque[b, :] = np.sum(aero.tangentialForce[:, b, :] * wtg.bladeData["r"][:, None] * dr[:, None], axis=0)

loads = {
    "F_normal": F_normal,
    "F_tangent": F_tangent,
    "Thrust": Thrust,
    "Torque": Torque,
}

# Dump results to temporary CSV
output_path = Path("Tmp/debug_results1.csv")
# Store everything into dictionary first
results_dict = {
    "position_x": sim.position[0].reshape(-1),
    "position_y": sim.position[1].reshape(-1),
    "position_z": sim.position[2].reshape(-1),
    "windSpeed_x": sim.windSpeed[0].reshape(-1),
    "windSpeed_y": sim.windSpeed[1].reshape(-1),
    "windSpeed_z": sim.windSpeed[2].reshape(-1),
    "AoA": aero.AoA.reshape(-1),
    "flowAngle": aero.flowAngle.reshape(-1),
    "lift": aero.lift.reshape(-1),
    "drag": aero.drag.reshape(-1),
    "normalForce": aero.normalForce.reshape(-1),
    "tangentialForce": aero.tangentialForce.reshape(-1),
    "inducedWind_y": aero.inducedWind[1].reshape(-1),
    "inducedWind_z": aero.inducedWind[2].reshape(-1),
    "F_normal": F_normal.reshape(-1),
    "F_tangent": F_tangent.reshape(-1),
    "Thrust": Thrust.reshape(-1),
    "Torque": Torque.reshape(-1),
}
# df = pd.DataFrame(results_dict, columns=results_dict.keys(), index=None)
# df.to_csv(output_path, index=False)

results = "Stored"
# ------------------------
# PLOT RESULTS
# ------------------------
plt.plot(sim.position[1, 0, 0, 1:], sim.position[0, 0, 0, 1:], label="Blade tip path")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Blade tip trajectory")
plt.grid()
plt.legend()
plt.show()


plt.figure()
plt.plot(sim.theta[0, 1:]/(np.pi*2), sim.windSpeed[2, 0, 0, 1:], label=r'Vz')
plt.plot(sim.theta[0, 1:]/(np.pi*2), sim.windSpeed[1, 0, 0, 1:], label=r'Vy')
plt.legend()
plt.xlabel("Revolutions")
plt.ylabel("Wind Speed (m/s)")
plt.grid()
plt.show()

