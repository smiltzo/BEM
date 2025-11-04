import numpy as np
from BEM_dataclasses import WTG, Wind, Simulation, AeroData
from typing import List
from scipy.interpolate import RegularGridInterpolator

# –––––––––––––––––––––––
# Reference systems
# –––––––––––––––––––––––
def get_a23(theta):
    return np.array([
        [ np.cos(theta),  -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [ 0, 0, 1]
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

def rpm_to_rads(rpm: float) -> float:
    return rpm * 2 * np.pi / 60.0

def rads_to_rpm(rads: float) -> float:
    return rads * 60.0 / (2 * np.pi)


# –––––––––––––––––––––––
# Flow
# –––––––––––––––––––––––
def wind_shear(wind: Wind, wtg: WTG, x: float):
    return wind.V0Ref * (x / wtg.H) ** wind.alpha if wind.alpha > 0 else wind.V0Ref


def check_element_in_tower(wtg: WTG, x: float) -> bool:
    return True if x > wtg.H else False


def tower_model(wind: Wind, wtg: WTG, y: float, z: float, eps: float = 1e-6):
    r = np.hypot(y, z)
    c, s = z/r, y/r  
    
    Vr = wind.Vz * (1 - (wtg.towerRadius / r)**2) * c
    Vt = wind.Vz * (1 + (wtg.towerRadius / r)**2) * s

    Vy =  (y/r) * Vr - (z/r) * Vt
    Vz =  (z/r) * Vr + (y/r) * Vt
    Vx =  0.0

    is_stagnant = (abs(Vx) < eps and abs(Vy) < eps and abs(Vz) < eps)
    return np.array([[Vx, Vy, Vz]]).T, is_stagnant


def flow_angle(v_axial, v_tangential) -> float:
    return np.arctan2(v_axial, v_tangential)


# –––––––––––––––––––––––
# CL and CD interpolation functions
# –––––––––––––––––––––––

def interpolate_lift(aero: AeroData, wtg: WTG,
                     point: tuple[float, float] | List[float] | np.ndarray) -> float: 
    Cl_interpolator = RegularGridInterpolator(
        (aero.airfoils["AoA"], wtg.thicknesses / 10),
        aero.airfoils["Cl"],
        bounds_error=False,
        fill_value=None)
    
    return float(Cl_interpolator(point))


def interpolate_coeffs_dyn_stall(aero: AeroData, wtg: WTG,
                               point: tuple[float, float] | List[float] | np.ndarray) -> tuple[float, float, float]:
    interp_cl_inv = RegularGridInterpolator(
        (aero.airfoils["AoA"], wtg.thicknesses / 10),
        aero.airfoils["Cl_inv"],
        bounds_error=False,
        fill_value=None)
    
    interp_fs = RegularGridInterpolator(
        (aero.airfoils["AoA"], wtg.thicknesses / 10), 
        aero.airfoils["f_s"],
        bounds_error=False,
        fill_value=None)
    
    interp_cl_fs = RegularGridInterpolator(
        (aero.airfoils["AoA"], wtg.thicknesses / 10), 
        aero.airfoils["Cl_fs"],
        bounds_error=False,
        fill_value=None)

    return (float(interp_cl_inv(point)), float(interp_fs(point)), float(interp_cl_fs(point)))


def interpolate_drag(aero: AeroData, wtg: WTG,
                     point: tuple[float, float] | List[float] | np.ndarray) -> float:
    Cd_interpolator = RegularGridInterpolator(
        (aero.airfoils["AoA"], wtg.thicknesses / 10), 
        aero.airfoils["Cd"],
        bounds_error=False,
        fill_value=None)
    
    return float(Cd_interpolator(point))


