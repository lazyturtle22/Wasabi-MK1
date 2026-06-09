"""
filters.py - Low-pass filters to smooth servo angle signals.

Two strategies are provided:
  - ExponentialSmoothing : lightweight, single-parameter, zero memory overhead.
  - MovingAverageFilter  : equal-weight window, trades latency for stability.

Use ExponentialSmoothing (alpha ≈ 0.20–0.30) for real-time teleoperation;
switch to MovingAverageFilter if your camera produces bursty dropout spikes.
"""

import collections
import numpy as np


class ExponentialSmoothing:
    """
    Exponential Moving Average (EMA) low-pass filter — one alpha per channel.

    Formula:  y[t] = alpha * x[t]  +  (1 - alpha) * y[t-1]

    alpha → 1 : tracks input instantly (more servo jitter)
    alpha → 0 : heavily damped (sluggish but very stable)
    """

    def __init__(self, alpha: float = 0.20, num_channels: int = 6):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.num_channels = num_channels
        self._state: np.ndarray | None = None

    def update(self, new_values: list[float]) -> list[float]:
        """Feed one sample; returns the smoothed values for all channels."""
        arr = np.asarray(new_values, dtype=float)
        if self._state is None:
            self._state = arr.copy()
        else:
            self._state = self.alpha * arr + (1.0 - self.alpha) * self._state
        return self._state.tolist()

    def reset(self) -> None:
        """Clear filter state (e.g., after a tracking dropout)."""
        self._state = None


class MovingAverageFilter:
    """
    Simple sliding-window moving average.

    Larger 'window' reduces jitter at the cost of added latency
    (latency ≈ window / 2 frames).
    """

    def __init__(self, window: int = 8, num_channels: int = 6):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._bufs = [collections.deque(maxlen=window) for _ in range(num_channels)]

    def update(self, new_values: list[float]) -> list[float]:
        smoothed = []
        for buf, val in zip(self._bufs, new_values):
            buf.append(val)
            smoothed.append(sum(buf) / len(buf))
        return smoothed

    def reset(self) -> None:
        for buf in self._bufs:
            buf.clear()
