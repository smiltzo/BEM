from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path

@dataclass
class WTG:

    # Lengths
    H: float = 119.0      # Hub height (m)
    Ls: float = 7.1     # Shaft length (m)
    R: float = 89.15    # Blade length (m)
    towerRadius: float = 3.32 # (m)
    blades: int = 3
    thicknesses: np.ndarray = field(
        default_factory=lambda: np.array([999, 600, 480, 360, 301, 241])
    )
    # Angles
    yaw: float = np.deg2rad(20.0)
    tilt: float = np.deg2rad(0.0)
    roll: float = np.deg2rad(0.0)
    cone: float = np.deg2rad(0.0)
    pitch0: float = np.deg2rad(2.0)


    bladeData: Dict = field(init=False)

    def __post_init__(self):
        df = pd.read_csv("Data/bladedat.csv", sep=',', header=0)
        self.bladeData = {col: np.array(values) for col, values in df.to_dict(orient='list').items()}


    def update_tower_radius(self, r: float) -> None:
        if r > self.H:
            self.towerRadius = 0.0
        else:
            self.towerRadius = 3.32


@dataclass
class Wind:

    V0_x: float = 0.0
    V0_y: float = 0.0
    V0_z: float = 9.0

    # Instantaneous wind components in ground system
    Vx: float = 0.0
    Vy: float = 0.0
    Vz: float = 9.0

    # Atmosphere
    density: float = 1.225 # kg/m^3
    hasShear: bool = True
    hasTowerEffect: bool = True
    alpha: float = 0.141   # (-)
    V0Ref: float = 10.0 # m/s



@dataclass
class Simulation:
    """Simulation parameters and time/state arrays for multi-blade FEM/BEM solver."""
    omega: float = 0.62        # Rotor angular velocity (rad/s)
    dt: float = 0.15
    duration: float = 300
    r: float = 70.0            # Element position (m)
    dynamicStall: bool = True

    # Some integers
    nBlades: int = 3
    bladeElements: int = 18

    # Derived arrays
    nSteps: int = field(init=False)
    times: np.ndarray = field(init=False)

    theta: np.ndarray = field(init=False)       # shape (nBlades, nSteps+1)
    position: np.ndarray = field(init=False)    # shape (3, bladeElements, nBlades, nSteps+1)
    windSpeed: np.ndarray = field(init=False)   # shape (3, bladeElements, nBlades, nSteps+1)

    dynamicInduction: float = field(init=False)



    def __post_init__(self):
        self.nSteps = int(self.duration / self.dt)
        self.times = np.linspace(0, self.duration, self.nSteps + 1)

        # Blade angles
        self.theta = np.zeros((self.nBlades, self.nSteps + 1))

        # Arrays for space + time
        self.position = np.zeros((3, self.bladeElements, self.nBlades, self.nSteps + 1))
        self.windSpeed = np.zeros_like(self.position)

        self.dynamicInduction = 0

    
    def update_induction_dyn_wake(self, value: float) -> None:
        if value > 0.5:
            self.dynamicInduction = 0.5
        else:
            self.dynamicInduction = value



@dataclass
class AeroData:
    """ Aerodynamic data and results """
    # raise NotImplementedError("Not ready yet")

    wtg: "WTG"
    sim: "Simulation"

    AXIAL = 2
    TANGENTIAL = 1

    airfoilFiles: List[Path] = field(default_factory=list)

    nBlades: int = field(init=False)
    nElements: int = field(init=False)
    nSteps: int = field(init=False)

    # Arrays to store aero quantities
    windSpeed: np.ndarray = field(init=False)
    inducedWind: np.ndarray = field(init=False)
    inducedWindQS: np.ndarray = field(init=False)
    inducedWindInt: np.ndarray = field(init=False)
    relWindSpeed: np.ndarray = field(init=False)

    flowAngle: np.ndarray = field(init=False)
    AoA: np.ndarray = field(init=False)
    lift: np.ndarray = field(init=False)
    drag: np.ndarray = field(init=False)

    normalForce: np.ndarray = field(init=False)
    tangentialForce: np.ndarray = field(init=False)

    separationFactor: np.ndarray = field(init=False)

    # Dynamic Wake induction time constants
    # tau1: float = field(init=False)
    # tau2: float = field(init=False)

    # pitch = np.ndarray = field(init=False)

    airfoils: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):

        # Get data from WTG and Simulation
        self.nBlades = self.wtg.blades
        self.nElements = self.sim.bladeElements
        self.nSteps = self.sim.nSteps + 1

        # Initialize aerodynamic arrays
        vector_shape = (3, self.nElements, self.nBlades, self.nSteps)
        induced_shape = (2, self.nElements, self.nBlades, self.nSteps)
        scalar_shape = (self.nElements, self.nBlades, self.nSteps)

        self.windSpeed = np.zeros(vector_shape)
        self.inducedWind = np.zeros(induced_shape)
        self.inducedWindQS = np.zeros(induced_shape)
        self.inducedWindInt = np.zeros(induced_shape)
        self.relWindSpeed = np.zeros(vector_shape)

        self.AoA = np.zeros(scalar_shape)
        self.lift = np.zeros(scalar_shape)
        self.drag = np.zeros(scalar_shape)
        self.flowAngle = np.zeros(scalar_shape)

        self.normalForce = np.zeros(scalar_shape)
        self.tangentialForce = np.zeros(scalar_shape)


        if self.sim.dynamicStall:
            self.separationFactor = np.zeros(scalar_shape)

        try:
            # Read all files and combine them into one DataFrame (side-by-side)
            dfs = [pd.read_csv(f"Data/{f}", sep=",", header=0) for f in self.airfoilFiles]
            df_all = pd.concat(dfs, axis=1, ignore_index=True)

            _AoA      = dfs[0].iloc[:, 0].to_numpy()  
            _Cl       = df_all.iloc[:, 1::7].to_numpy()  # every 7th column starting from 1
            _Cd       = df_all.iloc[:, 2::7].to_numpy()
            _Cm       = df_all.iloc[:, 3::7].to_numpy()
            _f_s      = df_all.iloc[:, 4::7].to_numpy()
            _Cl_inv   = df_all.iloc[:, 5::7].to_numpy()
            _Cl_fs    = df_all.iloc[:, 6::7].to_numpy()

        except Exception as e:
            print(f"Could not load airfoil data:\n\t{e}")

        self.airfoils = {}
        keys = ["AoA", "Cl", "Cd", "Cm", "f_s", "Cl_inv", "Cl_fs"]
        self.airfoils = {key: array for key, array in zip(keys, [_AoA, _Cl, _Cd, _Cm, _f_s, _Cl_inv, _Cl_fs])}

