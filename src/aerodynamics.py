import numpy as np 
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dataclassesBEM import WTG, Simulation

class AeroData:

    AXIAL = 2
    TANGENTIAL = 1

    def __init__(self, wtg: WTG, sim: Simulation) -> None:

        self.wtg = wtg
        self.sim = sim
        # self.airfoilFiles = airfoilFiles

        self.nBlades = self.wtg.blades
        self.nElements = self.sim.bladeElements
        self.nSteps = self.sim.nSteps + 1

        self.windSpeed = None
        self.inducedWind = None
        self.inducedWindQS = None
        self.inducedWindInt = None
        self.relWindSpeed = None

        self.AoA = None
        self.lift = None
        self.drag = None
        self.flowAngle = None

        self.Cl = None
        self.Cd = None
        self.cl_inv = None
        self.f_s = None
        self.cl_fs = None
        self.separationFactor = None

        self.normalForce = None
        self.tangentialForce = None

        self.airfoils: Dict[str, np.ndarray] = {}


    def allocate_arrays(self) -> None:

        vector_shape = (3, self.nElements, self.nBlades, self.nSteps)
        induced_shape = (2, self.nElements, self.nBlades, self.nSteps)
        scalar_shape = (self.nElements, self.nBlades, self.nSteps)

        self.windSpeed = np.zeros(vector_shape)
        self.inducedWind = np.zeros(vector_shape)
        self.inducedWindQS = np.zeros(vector_shape)
        self.inducedWindInt = np.zeros(induced_shape)
        self.relWindSpeed = np.zeros(vector_shape)

        self.AoA = np.zeros(scalar_shape)
        self.lift = np.zeros(scalar_shape)
        self.drag = np.zeros(scalar_shape)
        self.flowAngle = np.zeros(scalar_shape)

        self.Cl = np.zeros(scalar_shape)
        self.Cd = np.zeros(scalar_shape)

        if self.sim.hasDynamicStall:
            self._allocate_dynamic_stall(scalar_shape)

        self.normalForce = np.zeros(scalar_shape)
        self.tangentialForce = np.zeros(scalar_shape)


    def _allocate_dynamic_stall(self, shape):
        self.separationFactor = np.zeros(shape)
        self.cl_inv = np.zeros(shape)
        self.f_s = np.zeros(shape)
        self.cl_fs = np.zeros(shape)

    def load_airfoils(self, airfoilFiles: List[Path]) -> None:
        
        if not airfoilFiles:
            raise ValueError("No airfoil files provided")
        try:
            dfs = [pd.read_csv(f"Data/Airfoils/{f}", sep=',', header=0) for f in airfoilFiles]
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
