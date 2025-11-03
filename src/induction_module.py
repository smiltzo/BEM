import numpy as np
from BEM_dataclasses import WTG, Wind, Simulation, AeroData



# Aerodynamics
def compute_glauert_correction(aero: AeroData, sim: Simulation,
                               indices: tuple[int, int, int], use_prev=True, eps=1e-5) -> float:
    e, b, t = indices
    tt = t-1 if use_prev else t
    a = -aero.inducedWind[aero.AXIAL, e, b, t] / sim.windSpeed[aero.AXIAL, e, b, t]

    if a <= 1/3:
        return 1.0
    else:
        return 0.25*(5.0 - 3.0*a)
    

def compute_corrected_vel(aero: AeroData, sim: Simulation,
                          indices: tuple[int, int, int], use_prev=False) -> float:
    e, b, t = indices
    tt = t-1 if use_prev else t
    fg = compute_glauert_correction(aero, sim, indices, use_prev=use_prev)
    return np.hypot(sim.windSpeed[aero.TANGENTIAL, e, b, tt],
                    (sim.windSpeed[aero.AXIAL, e, b, tt] + fg * aero.inducedWind[aero.AXIAL, e, b, tt]))


def prandtl_tip_loss(wtg: WTG, aero: AeroData, indices: tuple[int, int, int]) -> float:
    e, j, i = indices

    phi = aero.flowAngle[*indices]
    sin_phi = np.sin(abs(phi))

    arg = np.exp((-wtg.blades / 2) * (wtg.R - wtg.bladeData["r"][e]) / (wtg.bladeData["r"][e] * sin_phi))

    # arg = np.clip(arg, 1e-4, 1-1.5e-4)

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
    Vz = aero.relWindSpeed[aero.AXIAL, e, b, t]
    eps = 1e-5

    Wqs = compute_quasi_steady_induction(aero, wtg, wind, sim, idx)
    aero.inducedWindQS[1:, e, b, t] = Wqs.squeeze()   # [tangential, axial]

    # Quasi Steady components
    Wqs_tangential = Wqs[0]    # Tangential
    Wqs_axial = Wqs[1]      # Axial     

    # Induced vels at previous time step
    Wqs_axial_prev = aero.inducedWindQS[aero.AXIAL, e, b, t - 1]    # Quasi steady
    W_int_axial_prev = aero.inducedWindInt[1, e, b, t - 1]                     # Integrated
    W_axial_prev = aero.inducedWind[aero.AXIAL, e, b, t - 1]          # Actual induced

    Wqs_tangt_prev = aero.inducedWindQS[aero.TANGENTIAL, e, b, t - 1]
    W_tangt_int_prev = aero.inducedWindInt[0, e, b, t - 1]
    W_tangt_prev = aero.inducedWind[aero.TANGENTIAL, e, b, t - 1]


    # Avoid dividing by zero for Vz
    dynInduction = -aero.inducedWind[aero.AXIAL, e, b, t] / Vz
    dynInduction = np.clip(dynInduction, 0.0, 0.5) 

    # Time constants
    tau1 = 1.1 / (1.0 - 1.3 * dynInduction) * wtg.R / Vz
    tau2 = (0.39 - 0.26 * (wtg.bladeData["r"][e] / wtg.R)**2) * tau1

    # Right Hand side of the ODE
    rhs_axial = Wqs_axial + k * tau1 * (Wqs_axial - Wqs_axial_prev) / sim.dt
    rhs_tangt = Wqs_tangential + k * tau1 * (Wqs_tangential - Wqs_tangt_prev) / sim.dt

    # ANalystical solutions
    W_axial_int = rhs_axial + (W_int_axial_prev - rhs_axial) * np.exp(-sim.dt / tau1)
    W_axial = W_axial_int + (W_axial_prev - W_axial_int) * np.exp(-sim.dt / tau2)

    W_tangt_int = rhs_tangt + (W_tangt_int_prev - rhs_tangt) * np.exp(-sim.dt / tau1)
    W_tangt = W_tangt_int + (W_tangt_prev - W_tangt_int) * np.exp(-sim.dt / tau2)

    aero.inducedWindInt[1, *idx] = W_axial_int
    aero.inducedWind[0, *idx] = W_tangt_int

    aero.inducedWind[aero.AXIAL, *idx] = W_axial
    aero.inducedWind[aero.TANGENTIAL, *idx] = W_tangt