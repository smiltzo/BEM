import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import BEM_utils as func
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

            if wind.hasTowerEffect:
                wtg.update_tower_radius(sim.position[0, *idx1])
                wind.Vz = V_local[2, 0]
                V_local = func.tower_model(wind, wtg,
                                      sim.position[1, *idx1],
                                      sim.position[2, *idx1])

            sim.windSpeed[:, *idx1] = func.go_to_ground_system(a12, a23_blade1, a34,
                                                                     V_local).squeeze()
            
            # BEM stuff - Induced wind
            aero.relWindSpeed[1, *idx1] = (sim.windSpeed[1, *idx1] + aero.inducedWind[1, *idx] - sim.omega * wtg.bladeData["r"][e] * wtg.cone).squeeze()
            aero.relWindSpeed[2, *idx1] = (sim.windSpeed[2, *idx1] + aero.inducedWind[2, *idx]).squeeze()

            aero.flowAngle[*idx1] = func.flow_angle(aero.relWindSpeed[1, *idx1],
                                                          aero.relWindSpeed[2, *idx1])
            aero.AoA[*idx1] = aero.flowAngle[*idx1] - (wtg.bladeData["twist"][e] + wtg.pitch0)

            relWindSpeedMagnitude = np.sqrt(aero.relWindSpeed[1, *idx1]**2 + aero.relWindSpeed[2, *idx1]**2)
            
            # Read and interpolate lift and drag
            Cd = func.interpolate_drag(aero, wtg, (aero.AoA[*idx1], wtg.bladeData["thick"][e]))

            if sim.dynamicStall:
                # S. Øye dyn. stall model
                cl_inv, f_s, cl_fs = func.interpolate_lift_dyn_stall(aero, wtg,
                                                                     (aero.AoA[*idx1], wtg.bladeData["thick"][e]))
                tau = 4 * wtg.bladeData["chord"][e] / relWindSpeedMagnitude
                aero.separationFactor[*idx1] = f_s + (aero.separationFactor[*idx] - f_s) * np.exp(-sim.dt / tau)
                Cl = aero.separationFactor[*idx1] * cl_inv + (1 - aero.separationFactor[*idx1]) * cl_fs
            else:
                # Just steady-state Cl
                Cl = func.interpolate_lift(aero, wtg, (aero.AoA[*idx1], wtg.bladeData["thick"][e]))

            # Compute element Lift and Drag
            lift = 0.5 * wind.density * relWindSpeedMagnitude**2 * Cl * wtg.bladeData["chord"][e]
            drag = 0.5 * wind.density * relWindSpeedMagnitude**2 * Cd * wtg.bladeData["chord"][e]
            
            aero.lift[*idx1] = lift.squeeze()
            aero.drag[*idx1] = drag.squeeze()

            # Rotate them into rotor plane (by the pitch angle)
            pNormal = lift * np.cos(np.rad2deg(aero.flowAngle[*idx1])) + drag * np.sin(np.rad2deg(aero.flowAngle[*idx1]))
            pTangent = drag * np.sin(np.rad2deg(aero.flowAngle[*idx1])) - lift * np.cos(np.rad2deg(aero.flowAngle[*idx1]))
            
            aero.normalForce[*idx1] = pNormal.squeeze()
            aero.tangentialForce[*idx1] = pTangent.squeeze()

            # ---- DYNAMIC INFLOW ----
            # Compute Quasi-Steady induced winds
            # aero.inducedWindQS[1:, *idx1] = func.compute_quasi_steady_induction(aero, wtg, wind, sim, idx1).squeeze()
            func.compute_induction(aero, sim, wtg, wind, idx1)
            end_inner_loop = "Done"




calculations = "Done"
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

