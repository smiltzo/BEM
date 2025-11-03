import numpy as np
from BEM_dataclasses import WTG, Wind, Simulation, AeroData




# Aerodynamics
def compute_glauert_correction(aero: AeroData, sim: Simulation,
                               indices: tuple[int, int, int], use_prev=True, eps=1e-5) -> float:
    e,b,t = indices
    tt = t-1 if use_prev else t
    a = -aero.inducedWind[aero.AXIAL, e, b, tt] / max(abs(sim.windSpeed[aero.AXIAL, e, b, tt]), eps)
    a = np.clip(a, 0.0, 0.5)   # IMPORTANT: enforce 0 <= a <= 0.5 for dynamic wake
    if a <= 1/3:
        return 1.0
    return 0.25*(5.0 - 3.0*a)
    

def compute_corrected_vel(aero: AeroData, sim: Simulation,
                          indices: tuple[int, int, int], use_prev=True) -> float:
    e,b,t = indices
    tt = t-1 if use_prev else t
    fg = compute_glauert_correction(aero, sim, indices, use_prev=use_prev)
    return np.hypot(sim.windSpeed[aero.TANGENTIAL, e, b, tt],
                    (sim.windSpeed[aero.AXIAL, e, b, tt] + fg * aero.inducedWind[aero.AXIAL, e, b, tt]))


def prandtl_tip_loss(wtg: WTG, aero: AeroData, indices: tuple[int, int, int]) -> float:
    e, j, i = indices

    phi = np.deg2rad(aero.flowAngle[*indices])
    sin_phi = np.sin(phi)
    sin_phi = np.where(np.abs(sin_phi) < 1e-6, 1e-6, sin_phi)

    arg = np.exp((-wtg.blades / 2) * (wtg.R - wtg.bladeData["r"][e]) /
                 (wtg.bladeData["r"][e] * sin_phi))

    arg = np.clip(arg, 1e-4, 1-1.5e-4)

    return 2 / np.pi * np.arccos(arg)



def compute_quasi_steady_induction(aero: AeroData, wtg: WTG, wind: Wind, sim: Simulation,
                                   idx: tuple[int, int, int]) -> np.ndarray:
    e, b, t = idx
    F = prandtl_tip_loss(wtg, aero, idx)
    V = compute_corrected_vel(aero, sim, idx)

    phi = aero.flowAngle[*idx]
    L = aero.lift[*idx]

    # Z -> Axial, Y -> Tangential
    Wz = float(-wtg.blades) * L * np.cos(phi) / (4 * wind.density * np.pi * wtg.bladeData["r"][e] * F * V)
    Wy = float(-wtg.blades) * L * np.sin(phi) / (4 * wind.density * np.pi * wtg.bladeData["r"][e] * F * V)

    return np.array([Wy, Wz])



def compute_induction(aero: AeroData, sim: Simulation, wtg: WTG, wind: Wind, idx: tuple[int,int,int], k: float = 0.6):
    e, b, t = idx
    Vz = wind.V0_z
    eps = 1e-5

    Wqs = compute_quasi_steady_induction(aero, wtg, wind, sim, idx)
    aero.inducedWindQS[:, e, b, t] = Wqs   # [tangential, axial]

    # Quasi Steady components
    Wqs_axial = Wqs[aero.AXIAL]
    Wqs_tangential = Wqs[aero.TANGENTIAL]

    # Induced vels at previous time step
    Wqs_axial_prev = aero.inducedWindQS[aero.AXIAL, e, b, t - 1]    # Quasi steady
    W_int_prev = aero.inducedWindInt[e, b, t - 1]             # Integrated
    W_axial_prev = aero.inducedWind[aero.AXIAL, e, b, t - 1]        # Actual induced


    # Avoid dividing by zero for Vz
    dynInduction = -aero.inducedWind[aero.AXIAL, e, b, t] / max(abs(Vz), eps)
    dynInduction = np.clip(dynInduction, 0.0, 0.5) 

    # Time constants
    tau1 = 1.1 / (1.0 - 1.3 * dynInduction) * wtg.R / max(Vz, eps)
    tau2 = (0.39 - 0.26 * (wtg.bladeData["r"][e] / wtg.R)**2) * tau1

    # Right Hand side of the ODE
    rhs = Wqs_axial + k * tau1 * (Wqs_axial - Wqs_axial_prev) / sim.dt

    # ANalystical solutions
    W_int = rhs + (W_int_prev - rhs) * np.exp(-sim.dt / tau1)
    W = W_int + (W_axial_prev - W_int) * np.exp(-sim.dt / tau2)

    aero.inducedWindInt[*idx] = W_int
    aero.inducedWind[aero.AXIAL, *idx] = W