from dataclasses import dataclass, field
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Optional, Tuple

@dataclass
class Wind:

    V0_x: float = 0.5
    V0_y: float = 0.5
    V0_z: float = 9.0

    # Instantaneous wind components
    Vx: float = V0_x
    Vy: float = V0_y
    Vz: float = V0_z

    wsp: Optional[np.ndarray] = None

    # Atmosphere
    density: float = 1.225 # kg/m^3
    isTurbulent: bool = True
    hasShear: bool = True
    hasTowerEffect: bool = True
    alpha: float = 0.141   # (-)
    V0Ref: float = 10.0 # m/s


    def update_wsp_time(self, time_idx: int) -> None:
        
        if self.wsp is None:
            self.Vx, self.Vy, self.Vz = self.V0_x, self.V0_y, self.V0_z
        else:
            self.Vx = self.wsp[0, time_idx]
            self.Vy = self.wsp[1, time_idx]
            self.Vz = self.wsp[2, time_idx]


@dataclass
class WindTurbulent(Wind):
    """Subclass that generates turbulence and fills Vx, Vy, Vz arrays."""

    # Turbulence box
    turbulence_box: Optional[np.ndarray] = None
    box_shape: Tuple[int, int, int] = (4096, 32, 32)                  # nx, ny, nz
    box_dimensions: Tuple[float, float, float] = (6142.5, 180.0, 180.0)  # Lx, Ly, Lz
    umean: float = Wind.V0_z  
    time: Optional[np.ndarray] = None                               # time array for box

    TI: float = None

    def load_mann_box(self, filename: str) -> None:
        """Load a Mann turbulence box from file."""
        n1, n2, n3 = self.box_shape
        data = np.fromfile(filename, np.dtype('<f'), -1)
        if len(data) != n1 * n2 * n3:
            raise ValueError(f"Data size {len(data)} does not match expected shape {self.box_shape}")
        self.turbulence_box = data.reshape(n1, n2, n3)

        # Compute time array
        Lx, Ly, Lz = self.box_dimensions
        deltax = Lx / (n1 - 1)
        deltat = deltax / self.umean
        self.time = np.arange(deltat, n1*deltat+deltat, deltat)
        print(f"Turbulence box loaded: shape {self.turbulence_box.shape}")


    def generate_turbulent_wind(self, y_idx: int = 16, z_idx: int = 16) -> None:
        """
        Return the wind array with the same components as the main code
            x: transversal
            y: lateral
            z: axial (main flow component)
        """
        if self.turbulence_box is None or self.time is None:
            raise ValueError("Turbulence box not loaded yet.")

        # Extract turbulent signal
        u_turb = self.turbulence_box[:, y_idx, z_idx]

        self.wsp = np.asarray([self.V0_x + np.zeros_like(u_turb),
                               self.V0_y + np.zeros_like(u_turb),
                               self.V0_z + u_turb])
        
        self.TI = np.std(u_turb) / self.umean * 100.0

        print(f"Generated Vx, Vy, Vz arrays with length {self.wsp.shape[1]}")


    def get_signal(self) -> dict:
        return {"Vx": self.Vx, "Vy": self.Vy, "Vz": self.Vz}
    

    def get_signal_at_point(self, y_idx: int, z_idx: int) -> np.ndarray:
        """Return the time series at a given spanwise/vertical point."""
        if self.turbulence_box is None:
            raise ValueError("Turbulence box not loaded yet.")
        return self.turbulence_box[:, y_idx, z_idx]
    

    def plot_contour_at_time(self, t_idx: int) -> None:
        """
        Plot a 2D contour of the turbulence plane at a given time index.
        """
        if self.turbulence_box is None:
            raise ValueError("Turbulence box not loaded yet.")
        
        fig, ax = plt.subplots()
        cp = ax.contourf(self.turbulence_box[t_idx, :, :])
        fig.colorbar(cp)
        ax.set_title(f'Turbulence contour at t index {t_idx}')
        plt.show()
    

    def plot_psd_at_point(self, y_idx: int, z_idx: int, nperseg: int = 1024) -> None:
        """
        Compute and plot the power spectral density at a given point.
        """
        sig = self.get_signal_at_point(y_idx, z_idx)

        if self.time is None:
            raise ValueError("Time array not initialized.")
        
        fs = 1 / (self.time[1] - self.time[0])
        f, Pxx_den = signal.welch(sig, fs, nperseg=nperseg)
        
        fig, ax = plt.subplots()
        ax.plot(f / 2*np.pi, Pxx_den, scaley='log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD')
        ax.set_title(f'PSD at point (y={y_idx}, z={z_idx})')
        plt.show()


