from dataclasses import dataclass, field
import numpy as np

@dataclass
class PitchControl:
    nSteps: int
    kp: float
    ki: float
    theta_dot_max_deg: float = 8.4           # deg/s (user-friendly)
    theta_init: float = 0.0                  # radians
    K1: float = np.deg2rad(14.0)             # radians
    K2: float = np.deg2rad(26.0)             # radians
    pitch_limits: tuple = (np.deg2rad(2.0), np.deg2rad(45.0))
    omega_ref: float = 1.06

    # internal state
    integral: float = 0.0
    pitch: np.ndarray = field(init=False)
    pitch_p: np.ndarray = field(init=False)
    pitch_i: np.ndarray = field(init=False)
    omega_hist: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pitch = np.zeros(self.nSteps + 1)
        self.pitch_p = np.zeros(self.nSteps + 1)
        self.pitch_i = np.zeros(self.nSteps + 1)
        self.omega_hist = np.zeros(self.nSteps + 1)
        self.pitch[0] = self.theta_init
        # convert to radians per second
        self.theta_dot_max = np.deg2rad(self.theta_dot_max_deg)
        self.integrator_limits = self.pitch_limits  # radians, tune as needed

    def step(self, i: int, omega: float, dt: float, debug=False) -> float:
        dt = max(dt, 1e-12)
        theta_curr = self.pitch[i]
        # GK uses current pitch
        denom = 1.0 + theta_curr / self.K1
        Gk = 1.0 / denom if denom != 0 and np.isfinite(denom) else 0.0
        Gk = float(np.clip(Gk, 1e-6, 1e3))

        # error positive when rotor faster than reference
        error = omega - self.omega_ref

        P_term = Gk * self.kp * error
        self.integral += Gk * self.ki * error * dt
        # anti-windup
        self.integral = float(np.clip(self.integral, *self.integrator_limits))
        I_term = self.integral

        pitch_cmd = P_term + I_term

        # rate limit
        max_delta = self.theta_dot_max * dt
        pitch_cmd = float(np.clip(pitch_cmd, theta_curr - max_delta, theta_curr + max_delta))
        # absolute limits
        pitch_cmd = float(np.clip(pitch_cmd, *self.pitch_limits))

        # store
        self.pitch_p[i+1] = P_term
        self.pitch_i[i+1] = I_term
        self.pitch[i+1] = pitch_cmd
        self.omega_hist[i] = omega

        if debug:
            print(f"[CTRL] i={i} Gk={Gk:.3g} P={P_term:.3g} I={I_term:.3g} pitch_cmd(deg)={np.rad2deg(pitch_cmd):.3f}")

        return pitch_cmd
