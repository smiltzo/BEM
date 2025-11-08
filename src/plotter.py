import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from dataclassesBEM import WTG, Simulation, AeroData, RotorForces
from wind import Wind


class Plotter:
    def __init__(self,
                 wtg: WTG, sim: Simulation,
                 aero: AeroData, wind: Wind, rotor: RotorForces,
                 style: str = 'default'):
        self.wtg = wtg
        self.sim = sim
        self.aero = aero
        self.wind = wind
        self.rotor = rotor
        self._get_all_quantities()

        styles = ['default', 'custom', 'custom-dark', 'seaborn', 'ggplot']
        if style not in styles:
            raise ValueError(f"Style '{style}' not recognized. Available styles: {styles}")
        plt.style.use(style)

    def _setup_figure(self, title: str, xlabel: str, ylabel: str, figsize=(8,4.5)) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig
    
    def _save_or_show(self, fig: plt.Figure, filename: Optional[str] = None,
                      show: bool = True, save: bool = False) -> None:
        if save and filename:
            fig.savefig(filename)
        if show:
            plt.show()
        # plt.close(fig)

    def _get_all_quantities(self):
        self.results = {
            # From Simulation dataclass
            "time": self.sim.times,
            "windSpeed": self.sim.windSpeed,

            # From RotorForces dataclass
            "Thrust": self.rotor.Thrust,
            "Torque": self.rotor.Torque,
            "Power": self.rotor.Power,
            "CT": self.rotor.CT,
            "CQ": self.rotor.CQ,
            "CP": self.rotor.CP,

            # From AeroData dataclass
            "relWindSpeed": self.aero.relWindSpeed,
            "flowAngle": np.rad2deg(self.aero.flowAngle),
            "AoA": self.aero.AoA,
            "inducedWind": self.aero.inducedWind,
            "inducedWindQS": self.aero.inducedWindQS,
            "inducedWindInt": self.aero.inducedWindInt,
            "lift": self.aero.lift,
            "drag": self.aero.drag,
            "Cd": self.aero.Cd,
            "Cl": self.aero.Cl,
            "Cl_inv": self.aero.cl_inv,
            "f_s": self.aero.f_s,
            "Cl_fs": self.aero.cl_fs,
            "normalForce": self.aero.normalForce,
            "tangentialForce": self.aero.tangentialForce,
        }

    def plot_timeseries(self, quantities: list[str],
                    figSpecs: dict,
                    indices: tuple | None = None,
                    save: bool = False, show: bool = True,
                    filename: Optional[str] = None) -> plt.Figure:
    
        if indices != None:
            if len(indices) == 3:
                element, blade, _ = indices
            elif len(indices) == 4:
                component, element, blade, _ = indices

        t = self.results["time"]

        fig = self._setup_figure(figSpecs.get("title", "Time Series"), "Time (s)",
                                 figSpecs.get("ylabel", "Value"))
        ax = fig.axes[0]

        for quantity in quantities:
            if not quantity in self.results.keys():
                raise AttributeError(f"Results has no attribute '{quantity}'.\nChoose one of {list(self.results.keys())}.")

            data = self.results[quantity]
            dataDims = data.ndim

        
            if dataDims == 1:
                y = data[:]
            elif dataDims == 3:
                y = data[element, blade, :]
            elif dataDims == 4:
                y = data[component, element, blade, :]
            else:
                raise ValueError(f"Unsupported data dimension {dataDims} for quantity '{quantity}'.")

            ax.plot(t, y, label=quantity)

        ax.legend()
        ax.set_xlabel("Time (s)")
        self._save_or_show(fig, filename=filename, save=save, show=show)
        return fig
    

    def plot_spanwise(self, quantities: list[str],
                  figSpecs: dict,
                  indices: tuple[int, int, int, int],
                  save: bool = False, show: bool = True,
                  filename: Optional[str] = None) -> plt.Figure:
    
        if indices != None:
            if len(indices) == 3:
               _ , blade, time_idx = indices
            elif len(indices) == 4:
                component, _, blade, time_idx = indices

        fig = self._setup_figure(figSpecs.get("title", "Spanwise Distribution"),
                                 "Blade Length (m)",
                                 figSpecs.get("ylabel", "Value"))
        ax = fig.axes[0]

        for quantity in quantities:
            if not quantity in self.results.keys():
                raise AttributeError(f"Results has no attribute '{quantity}'.\nChoose one of {list(self.results.keys())}.")

            data = self.results[quantity]
            dataDims = data.ndim

            # Slice data according to its dimensions
            if dataDims == 1:
                y = data[:]  # just the 1D data
            elif dataDims == 3:
                y = data[:, blade, time_idx]  # spanwise (:) instead of element
            elif dataDims == 4:
                y = data[component, :, blade, time_idx]  # spanwise (:) in 4D
            else:
                raise ValueError(f"Unsupported data dimension {dataDims} for quantity '{quantity}'.")

            # x-axis: spanwise index
            x = np.linspace(0, self.wtg.R, y.shape[0])
            ax.plot(x, y, label=quantity)

        ax.legend()
        self._save_or_show(fig, filename=filename, save=save, show=show)
        return fig




