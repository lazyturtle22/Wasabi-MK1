"""
filters.py — Signal conditioning for real-time teleoperation.

OneEuroFilter
    The standard filter for human-motion teleop (Casiez et al., CHI 2012).
    Adaptive low-pass: heavy smoothing when the signal is slow (kills jitter),
    light smoothing when it moves fast (kills lag). Vastly better than a
    fixed-alpha EMA for this use case. Operates element-wise on numpy arrays,
    so one instance can filter a whole (N, 3) stack of landmarks.

SlewRateLimiter
    Hard cap on output velocity (rad/s). Applied to the final joint commands
    as a safety layer — essential once this drives real servos.
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """Element-wise One Euro filter over arbitrarily-shaped numpy arrays.

    Parameters
    ----------
    min_cutoff : baseline cutoff frequency in Hz. Lower = smoother when still.
    beta       : speed coefficient. Higher = snappier response to fast motion.
    d_cutoff   : cutoff for the internal derivative estimate (1 Hz is standard).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x

        dt = t - self._t
        if dt <= 1e-6:
            return self._x
        self._t = t

        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class SlewRateLimiter:
    """Clamp per-channel rate of change to max_rate (units/s)."""

    def __init__(self, max_rate: float):
        self.max_rate = float(max_rate)
        self._x: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x = x.copy()
            self._t = t
            return self._x

        dt = max(t - self._t, 1e-6)
        self._t = t
        step = np.clip(x - self._x, -self.max_rate * dt, self.max_rate * dt)
        self._x = self._x + step
        return self._x
