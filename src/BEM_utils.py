import numpy as np
from BEM_dataclasses import WTG, Wind, Simulation, AeroData
from typing import List
from scipy.interpolate import RegularGridInterpolator

# –––––––––––––––––––––––
# Reference systems
# –––––––––––––––––––––––
def get_a23(theta: float) -> np.ndarray:
    """
    Rotation matrix for azimuthal rotation of the blade (shaft to blade).
    """
    return np.array([
        [np.cos(theta),  np.sin(theta),  0.0],
        [-np.sin(theta), np.cos(theta),  0.0],
        [0.0,             0.0,           1.0]
    ])


    
def get_position(wtg: WTG, sim: Simulation,
                 a12: np.ndarray = np.eye(3),
                 a23: np.ndarray = np.eye(3),
                 a34: np.ndarray = np.eye(3)) -> np.ndarray:
    """
    Computes the position of a point on the blade span in the tower coordinate system.
    """
    pos_tower = np.array([[wtg.H, 0.0, 0.0]]).T
    pos_shaft = a12.T @ np.array([[0.0, 0.0, -wtg.Ls]]).T
    pos_blade = (a34 @ a23 @ a12).T @ np.array([[sim.r, 0.0, 0.0]]).T

    return pos_tower + pos_shaft + pos_blade


def go_to_blade_system(a12: np.ndarray, a23: np.ndarray, a34: np.ndarray,
                        input_array: np.ndarray) -> np.ndarray:
    return (a34 @ a23 @ a12).T @ input_array


def go_to_ground_system(a12: np.ndarray, a23: np.ndarray, a34: np.ndarray,
                        input_array: np.ndarray) -> np.ndarray:
    return (a34 @ a23 @ a12) @ input_array


# –––––––––––––––––––––––
# Flow
# –––––––––––––––––––––––
def wind_shear(wind: Wind, wtg: WTG, x: float):
    return wind.V0Ref * (x / wtg.H) ** wind.alpha if wind.alpha > 0 else wind.V0Ref


def check_element_in_tower(wtg: WTG, x: float) -> bool:
    return True if x > wtg.H else False


def tower_model(wind: Wind, wtg: WTG, y: float, z: float):
    r = np.sqrt(y**2 + z**2)
    c, s = z/r, y/r  

    Vr = wind.Vz * (1 - (wtg.towerRadius / r)**2) * c
    Vt = wind.Vz * (1 + (wtg.towerRadius / r)**2) * s

    Vy =  (y/r) * Vr - (z/r) * Vt
    Vz =  (z/r) * Vr + (y/r) * Vt
    Vx =  0.0

    return np.array([[Vx, Vy, Vz]]).T


def flow_angle(vector1: float, vector2: float, returnDegrees: bool = True) -> float:
    """Returns the results of tan^-1(vector2/vector1) in degreed by default"""
    if returnDegrees:
        return np.rad2deg(np.arctan2(vector2, vector1))
    else:
        return np.arctan2(vector2, vector1)


# CL and CD interpolation functions
def interpolate_lift(aero: AeroData, wtg: WTG,
                     point: tuple[float, float] | List[float] | np.ndarray) -> float: 
    Cl_interpolator = RegularGridInterpolator(aero.airfoils["AoA"], wtg.thicknesses / 10, aero.airfoils["Cl"])
    return Cl_interpolator(point)


def interpolate_lift_dyn_stall(aero: AeroData, wtg: WTG,
                               point: tuple[float, float] | List[float] | np.ndarray) -> tuple[float, float, float]:
    interp_cl_inv = RegularGridInterpolator((aero.airfoils["AoA"], wtg.thicknesses / 10), aero.airfoils["Cl_inv"])
    interp_fs = RegularGridInterpolator((aero.airfoils["AoA"], wtg.thicknesses / 10), aero.airfoils["Cl_fs"])
    interp_cl_fs = RegularGridInterpolator((aero.airfoils["AoA"], wtg.thicknesses / 10), aero.airfoils["Cl_fs"])

    return (interp_cl_inv(point), interp_fs(point), interp_cl_fs(point))


def interpolate_drag(aero: AeroData, wtg: WTG,
                     point: tuple[float, float] | List[float] | np.ndarray) -> float:
    Cd_interpolator = RegularGridInterpolator((aero.airfoils["AoA"], wtg.thicknesses / 10), aero.airfoils["Cd"])
    return Cd_interpolator(point)


# Aerodynamics
def compute_glauert_correction(aero: AeroData, sim: Simulation, indices: tuple[int, int, int]) -> float:
    element, blade, time = indices
    inductionFactor = -aero.inducedWind[2, *indices] / abs(sim.windSpeed[2, *indices])

    # Update the induction factor such that it does not exceed 0.5 for the dynamic wake
    sim.update_induction_dyn_wake(inductionFactor)

    if inductionFactor <= 1/3:
        return 1.0
    else:
        return 1/4 * (5.0 - 3 * inductionFactor)
    

def compute_corrected_vel(aero: AeroData, sim: Simulation, indices: tuple[int, int, int]) -> float:
    """ Returns the value of |V0 + fg * Wn|"""
    element, blade, time = indices
    fg = compute_glauert_correction(aero, sim, indices)
    lhs = np.sqrt(sim.windSpeed[1,  *indices]**2 + \
                  (sim.windSpeed[2, *indices] + fg * aero.inducedWind[2, element, blade, time - 1])**2)
    return lhs


def prandtl_tip_loss(wtg: WTG, aero: AeroData, indices: tuple[int, int, int]) -> float:
    e, j, i = indices

    phi = np.deg2rad(aero.flowAngle[*indices])
    sin_phi = np.sin(phi)
    sin_phi = np.where(np.abs(sin_phi) < 1e-6, 1e-6, sin_phi)

    arg = np.exp((-wtg.blades / 2) * (wtg.R - wtg.bladeData["r"][e]) /
                 (wtg.bladeData["r"][e] * sin_phi))

    arg = np.clip(arg, 0.0, 1.0)

    return 2 / np.pi * np.arccos(arg)



def compute_quasi_steady_induction(aero: AeroData, wtg: WTG, wind: Wind, sim: Simulation,
                                   idx: tuple[int, int, int]) -> np.ndarray:
    e, b, t = idx
    F = prandtl_tip_loss(wtg, aero, idx)
    V = compute_corrected_vel(aero, sim, idx)

    Wz = float(-wtg.blades) * aero.lift[*idx] * np.cos(np.deg2rad(aero.flowAngle[*idx])) / (4 * wind.density * np.pi * wtg.bladeData["r"][e] * F * V)
    Wy = float(-wtg.blades) * aero.lift[*idx] * np.sin(np.deg2rad(aero.flowAngle[*idx])) / (4 * wind.density * np.pi * wtg.bladeData["r"][e] * F * V)

    return np.array([[Wy, Wz]]).T





def compute_induction(aero: AeroData, sim: Simulation, wtg: WTG, wind: Wind,
                      idx: tuple[int, int, int], k: float = 0.6) -> np.ndarray:
    e, b, t = idx
    Vz = aero.windSpeed[2, e, b, t]

    aero.inducedWindQS[1:, *idx] = compute_quasi_steady_induction(aero, wtg, wind, sim, idx).squeeze()

    Wqs = aero.inducedWindQS[2, e, b, t]
    Wqs_prev = aero.inducedWindQS[2, e, b, t - 1]

    induction = -aero.inducedWind[2, e, b, t] / abs(Vz)
    sim.update_induction_dyn_wake(induction)
    
    tau1 = 1.1 / (1.0 - 1.3 * sim.dynamicInduction) * wtg.R / Vz
    tau2 = (0.39 - 0.26 * (wtg.bladeData["r"][e] / wtg.R)**2) * tau1

    rhs = Wqs + k * tau1 * (Wqs - Wqs_prev) / sim.dt

    W_int = rhs + (aero.inducedWindInt[e, b, t - 1] - rhs) * np.exp(-sim.dt/tau1)
    W = W_int + (aero.inducedWind[2, e, b, t - 1] - W_int) * np.exp(-sim.dt/tau2)

    # Assign them
    aero.inducedWindInt[*idx] = W_int
    aero.inducedWind[2, *idx] = W

