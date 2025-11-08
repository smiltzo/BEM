import numpy as np
import utils as func
import induction as ind
from dataclassesBEM import WTG, Simulation, RotorForces
from aerodynamics import AeroData
from controller import PitchControl
from  wind import WindTurbulent


# np.seterr(all='raise')
# warnings.filterwarnings("error")

# Dataclasses
wtg = WTG(
    pitch0=np.deg2rad(25.0),
    yaw=0.0
)
sim = Simulation(
    duration = 100.0,
    hasDynamicStall=True
)
aero = AeroData(wtg, sim)
aero.allocate_arrays()
aero.load_airfoils([f"FFA-W3-{n}.csv" for n in wtg.thicknesses])


rotor = RotorForces(sim)

tangt = aero.TANGENTIAL
axial = aero.AXIAL


# Load turbulence box
wind = WindTurbulent(
    isTurbulent=True,
    hasShear = True,
    hasTowerEffect = True,
    V0_z=13.0
)

if wind.isTurbulent:
    wind.load_mann_box("Data/Turb/sim1.bin")
    wind.generate_turbulent_wind(y_idx=16, z_idx=16)
    # wind.plot_contour_at_time(150)
    # wind.plot_psd_at_point(y_idx=16, z_idx=16)

ctrl = PitchControl(
    nSteps = sim.nSteps,
    kp = 1.5,
    ki = 0.64
)
steadyPower = 0.5 * wind.density * np.pi * wtg.R**2 * wind.V0_z**3 * 16/27

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
    wind.update_wsp_time(i)
    if i == 341:
        stop = True

    if sim.doPitchStep:
        if sim.times[i] > 100.0 and sim.times[i] <= 150.0:
            wtg.pitch0 = 2.0
        elif sim.times[i] > 150.0:
            wtg.pitch0 = 0.0
    
    # BLADE LOOP
    for j in range(wtg.blades):

        sim.theta[j, i + 1] = sim.theta[j, i] + rotor.omega[i] * sim.dt
        a23_blade1 = func.get_a23(sim.theta[j, i + 1])

        # BLADE ELEMENT LOOP
        for e in range(sim.bladeElements - 1):
            idx1 = (e, j, i + 1)
            idx  = (e, j, i)

            sim.position[:, *idx1] = func.get_position(wtg, sim, a12, a23_blade1, a34).squeeze()
            V_local = np.array([[0.0, 0.0, wind.Vz]]).T

            if wind.hasShear:
                V_local[axial, 0] = func.wind_shear(wind, wtg, sim.position[0, *idx1]) # To be updated with wind turbulence


            if wind.hasTowerEffect:
                wtg.update_tower_radius(sim.position[0, *idx1])
                V_local, isStagnant = func.tower_model(wind, wtg,
                                      sim.position[tangt, *idx1],
                                      sim.position[axial, *idx1])  
                if isStagnant:
                    V_local = sim.windSpeed[:, *idx]

            sim.windSpeed[:, *idx1] = func.go_to_blade_system(a12, a23_blade1, a34,
                                                                     V_local).squeeze()
            # sim.windSpeed[: ,*idx1] = V_local.squeeze()

            # BEM stuff - Induced wind
            aero.relWindSpeed[tangt, *idx1] = (sim.windSpeed[tangt, *idx1] + aero.inducedWind[tangt, *idx] - rotor.omega[i] * wtg.bladeData["r"][e] * np.cos(wtg.cone)).squeeze()
            aero.relWindSpeed[axial, *idx1] = (sim.windSpeed[axial, *idx1] + aero.inducedWind[axial, *idx]).squeeze()

            try:
                flowAngle = np.arctan2(aero.relWindSpeed[axial, *idx1], -aero.relWindSpeed[tangt, *idx1])
            except ValueError as err:
                print(f"Error computing flow angle at idx {idx1}: {err}")
                print(f"Vrel_y: {aero.relWindSpeed[1, *idx1]}, Vrel_z: {aero.relWindSpeed[2, *idx1]}")
                flowAngle = 0.0

            aero.flowAngle[*idx1] = flowAngle

            # AoA in degrees
            aero.AoA[*idx1] = np.rad2deg(aero.flowAngle[*idx1]) - (wtg.bladeData["twist"][e] + np.rad2deg(ctrl.pitch[i]))

            relWindSpeedMagnitude = np.hypot(aero.relWindSpeed[tangt, *idx1], aero.relWindSpeed[axial, *idx1])

            # Read and interpolate lift and drag
            point = np.atleast_2d((aero.AoA[*idx1], wtg.bladeData["thick"][e]))
            Cd = func.interpolate_drag(aero, wtg, point)

            if sim.hasDynamicStall:
                # S. Øye dyn. stall model
                cl_inv, f_s, cl_fs = func.interpolate_coeffs_dyn_stall(aero, wtg, point)

                if idx == (0, 0, 0):
                    aero.separationFactor[:, 0, 0] = 0.1

                tau = 4 * wtg.bladeData["chord"][e] / relWindSpeedMagnitude
                aero.separationFactor[*idx1] = f_s + (aero.separationFactor[*idx] - f_s) * np.exp(-sim.dt / tau)
                Cl = aero.separationFactor[*idx1] * cl_inv + (1 - aero.separationFactor[*idx1]) * cl_fs

                aero.cl_inv[*idx1] = cl_inv
                aero.f_s[*idx1] = f_s
                aero.cl_fs[*idx1] = cl_fs

            else:
                # Just steady-state Cl
                Cl = func.interpolate_lift(aero, wtg, point)

            # Assign coefficients
            aero.Cl[*idx1] = Cl
            aero.Cd[*idx1] = Cd

            # Compute element Lift and Drag
            lift = 0.5 * wind.density * abs(relWindSpeedMagnitude)**2 * Cl * wtg.bladeData["chord"][e]
            drag = 0.5 * wind.density * abs(relWindSpeedMagnitude)**2 * Cd * wtg.bladeData["chord"][e]
            aero.lift[*idx1] = lift
            aero.drag[*idx1] = drag

            # Rotate them into rotor plane (by the pitch angle)
            pNormal =  lift * np.cos(aero.flowAngle[*idx1]) + drag * np.sin(aero.flowAngle[*idx1])
            pTangent = lift * np.sin(aero.flowAngle[*idx1]) - drag * np.cos(aero.flowAngle[*idx1])
            aero.normalForce[*idx1] = pNormal
            aero.tangentialForce[*idx1] = pTangent

            # ---- DYNAMIC INFLOW ----
            ind.compute_induction(
                aero, sim, wtg, wind,
                idx1
            )
            end_inner_loop = "Done"

    # PI controller
    ctrl.step(i=i, omega=rotor.omega[i], dt=sim.dt)

    # Compute rotor forces and torque
    Fn = aero.normalForce[:, :, i + 1]
    Ft = aero.tangentialForce[:, :, i + 1]


    T = np.trapz(np.sum(Fn, axis=1), wtg.bladeData["r"], axis=0)
    Q = np.trapz((np.sum(Ft, axis=1) * wtg.bladeData["r"]), wtg.bladeData["r"], axis=0)

    P = Q * rotor.omega[i]

    rotor.CT[i+1] = T / (0.5 * wind.density * np.pi * wtg.R**2 * sim.windSpeed[2, *idx1]**2)
    rotor.CQ[i+1] = Q / (0.5 * wind.density * np.pi * wtg.R**2 * sim.windSpeed[2, *idx1]**2 * wtg.R)
    rotor.CP[i+1] = P / (0.5 * wind.density * np.pi * wtg.R**2 * sim.windSpeed[2, *idx1]**3)

    rotor.Thrust[i+1], rotor.Torque[i+1], rotor.Power[i+1] = T, Q, P

    # CpMax = steadyPower / (0.5 * wind.density * np.pi * wtg.R**2 * wind.V0_z**3)
    K = 1 / 2 / rotor.TSR**3 * wind.density * np.pi * wtg.R**5 * 0.4619
    Pgen = K * rotor.omega[i]**3
    # omegaGen = (Pgen / K ) ** (1/3)

    if rotor.omega[i] <= ctrl.omega_ref:
        Mgen = K * rotor.omega[i]**2
    else:
        Mgen = wtg.ratedPower / rotor.omega[i]


    rotor.omega[i + 1] = rotor.omega[i] + (Q - Mgen) / rotor.momentInertia * sim.dt
    rotor.Mgen[i + 1], rotor.Pgen[i + 1] = Mgen, Pgen


calculations = "Done"
# Forces, moments and power calculation


# ------------------------
# PLOT RESULTS
# ------------------------
from plotter import Plotter
plot = Plotter(wtg, sim, aero, wind, rotor, style='ggplot')

figspace = {
    "title": "Thrust, Torque and Power over Time",
    "ylabel": "Thrust (N) / Torque (Nm) / Power (W)"
}
plot.plot_timeseries(["Thrust", "Torque", "Power"], figspace)
plot.plot_spanwise(["normalForce", "tangentialForce"],
                   {"title": r"Spanwise Pn and Pt", "ylabel": "Force per unit length (N/m)"}, (16, 0, 600))
plot.plot_timeseries(["normalForce", "tangentialForce"], {"title": r"Normal and Tangent forces", "ylabel": "Force per unit length (N/m)"}, (15, 0, None))
plot.plot_spanwise(["AoA", "flowAngle"],
                   {"title": r"Spanwise AoA and Flow Angle", "ylabel": "Angle (deg)"},  (None, 0, 600))

plot.plot_timeseries(["inducedWind"], {"title": r"Induced wind W_z", "ylabel": "Induced wind (m/s)"}, (2, 15, 0, None))
plot.plot_timeseries(["inducedWind"], {"title": r"Induced wind W_y", "ylabel": "Induced wind (m/s)"}, (1, 15, 0, None))

plot.plot_timeseries(["Cd"], {"title": r"Cd time history", "ylabel": "Cd (-)"}, (15, 0, None))
plots = "Plotted"

plot.plot_timeseries(["inducedWind", "inducedWindQS"],
                     {"title": r"Induced winds", "ylabel": "Induced wind (m/s)"},
                     (2, 14, 0, None))

plot.plot_timeseries(["inducedWind", "inducedWindQS"],
                     {"title": r"Induced winds", "ylabel": "Induced wind (m/s)"},
                     (1, 14, 0, None))

plot.plot_timeseries(["CP", "CT"], 
                     {"title": r"Power and Thrust Coefficients", "ylabel": "Coefficient (-)"},
                     )

plot.plot_spanwise(["inducedWind", "inducedWindQS"],
                   {"title": "Axial Spanwise induction", "ylabel": "Induction velocity (m/s)"},
                   (2, None, 0, 600))

plot.plot_spanwise(["inducedWind", "inducedWindQS"],
                   {"title": "Tangential Spanwise induction", "ylabel": "Induction velocity (m/s)"},
                   (1, None, 0, 600))

plot.plot_timeseries(["Power"], {"title": "Power", "ylabel": "Power (W)"})

plot.plot_timeseries(["AoA", "flowAngle"], {"title": r"\alpha and \phi", "ylabel": "Angle (deg)"}, (15, 0, None))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(12,7))
ax[0].plot(sim.times[1:], rotor.omegaGen[1:], label='controller')
ax[0].set_title('Controller')
ax[1].plot(sim.times[1:], rotor.omega[1:], label='rotor')
ax[1].hlines(y=ctrl.omega_ref, xmin=0, xmax=sim.times[-1], linestyles='--')
ax[1].set_title('Rotor')
fig.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 1, figsize=(12,7))
ax[0].plot(sim.times[1:], rotor.Mgen[1:], label='Generator')
ax[0].plot(sim.times[1:], rotor.Torque[1:], label='Aerodynamic')
ax[0].legend()
ax[0].set_title('Gen Moment')
ax[1].plot(sim.times[1:], rotor.Pgen[1:], label='Generator')
ax[1].plot(sim.times[1:], rotor.Power[1:], label='Aerodynamic')
ax[1].set_title('Power')
ax[1].legend()
fig.tight_layout()
plt.show()


plt.figure()
plt.plot(sim.times[1:], np.rad2deg(ctrl.pitch[1:]))
plt.xlabel('Time (s)')
plt.ylabel('Pitch angle (deg)')
plt.show()

