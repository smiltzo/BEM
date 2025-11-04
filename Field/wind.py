from dataclasses import dataclass, field
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Optional, Tuple

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
class WindTurbulent(Wind):
    """Wind subclass that adds turbulence handling using Mann boxes."""
    
    # Turbulence box data
    turbulence_box: Optional[np.ndarray] = None
    box_shape: Tuple[int,int,int] = (4096, 32, 32)  # (nx, ny, nz)
    box_dimensions: Tuple[float,float,float] = (6142.5, 180.0, 180.0)  # (Lx, Ly, Lz)
    umean: float = 9.0  # mean wind speed
    time: Optional[np.ndarray] = None  # time array for the box
    
    def load_mann_box(self, filename: str):
        """Load a Mann turbulence box from a binary file."""
        n1, n2, n3 = self.box_shape
        data = np.fromfile(filename, np.dtype('<f'), -1)
        if len(data) != n1 * n2 * n3:
            raise ValueError(f"Data size {len(data)} does not match box shape {self.box_shape}")
        self.turbulence_box = data.reshape(n1, n2, n3)
        
        # Compute time array
        Lx, Ly, Lz = self.box_dimensions
        deltax = Lx / (n1 - 1)
        deltay = Ly / (n2 - 1)
        deltax = Lx / (n1 - 1)
        deltat = deltax / self.umean
        self.time = np.arange(deltat, n1*deltat+deltat, deltat)
        print(f"Turbulence box loaded with shape {self.turbulence_box.shape}")
    
    def get_signal_at_point(self, y_idx: int, z_idx: int) -> np.ndarray:
        """Return the time series at a given spanwise/vertical point."""
        if self.turbulence_box is None:
            raise ValueError("Turbulence box not loaded yet.")
        return self.turbulence_box[:, y_idx, z_idx]
    
    def plot_contour_at_time(self, t_idx: int):
        """Plot a 2D contour of the turbulence plane at a given time index."""
        if self.turbulence_box is None:
            raise ValueError("Turbulence box not loaded yet.")
        fig, ax = plt.subplots()
        cp = ax.contourf(self.turbulence_box[t_idx, :, :])
        fig.colorbar(cp)
        ax.set_title(f'Turbulence contour at t index {t_idx}')
        plt.show()
    
    def plot_psd_at_point(self, y_idx: int, z_idx: int, nperseg: int = 1024):
        """Compute and plot the power spectral density at a given point."""
        sig = self.get_signal_at_point(y_idx, z_idx)
        if self.time is None:
            raise ValueError("Time array not initialized.")
        fs = 1 / (self.time[1] - self.time[0])
        f, Pxx_den = signal.welch(sig, fs, nperseg=nperseg)
        
        fig, ax = plt.subplots()
        ax.loglog(f, Pxx_den)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD')
        ax.set_title(f'PSD at point (y={y_idx}, z={z_idx})')
        plt.show()


p = "Data/Turb/sim1.bin"