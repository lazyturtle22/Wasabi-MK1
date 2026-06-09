"""
angle_mapper.py — Converts Wasabi-MK1 human servo angles (degrees) to
                  MuJoCo 3-DoF arm joint angles (radians).

Human arm convention (from kinematics.py):
  Base Pan     0–180°   0° = arm swept right, 90° = forward, 180° = arm swept left
  Shoulder     0–180°   0° = arm straight up,  90° = horizontal, 180° = arm down
  Elbow Tilt   0–180°   0° = fully bent,        180° = arm fully extended / straight

Robot joint convention (mujoco_sim.py):
  joint_base     [-π/2, π/2]  rad   (arm pan; 0 = centred)
  joint_shoulder [0,    π/2]  rad   (arm up = 0, arm horizontal = π/2)
  joint_elbow    [0,    2.27] rad   (arm straight = 0, fully bent ≈ 130° = 2.27 rad)
"""

from __future__ import annotations

import math
import numpy as np


# ── Per-joint safe clamping ranges (degrees, before conversion) ───────────────

_BASE_RANGE     = (-90.0,  90.0)   # ±90° pan from centre
_SHOULDER_RANGE = (  0.0,  90.0)   # raise arm from vertical to horizontal
_ELBOW_RANGE    = (  0.0, 130.0)   # 0 = straight, 130 = max bend


def human_to_robot(
    base_deg:     float,
    shoulder_deg: float,
    elbow_deg:    float,
) -> tuple[float, float, float]:
    """
    Map three human-arm angles to MuJoCo joint setpoints in radians.

    Parameters
    ----------
    base_deg     : Human base-pan angle  [0–180°]
    shoulder_deg : Human shoulder angle  [0–180°]
    elbow_deg    : Human elbow angle     [0–180°]

    Returns
    -------
    (base_rad, shoulder_rad, elbow_rad) — clamped to each joint's safe range.
    """
    # Base: human 90° = arm forward → robot 0° (centred).  Linear map ±90°.
    base_mapped = base_deg - 90.0
    base_rad    = math.radians(
        float(np.clip(base_mapped, *_BASE_RANGE))
    )

    # Shoulder: human 0° (arm up) → robot 0°; human 90° (horizontal) → π/2.
    # Clamp to [0, 90°] — arm is not driven below horizontal.
    shoulder_rad = math.radians(
        float(np.clip(shoulder_deg, *_SHOULDER_RANGE))
    )

    # Elbow: human 180° (straight) → robot 0°; human 0° (bent) → 130° (2.27 rad).
    # Invert: more-bent human elbow → larger positive robot joint angle.
    elbow_mapped = 180.0 - elbow_deg
    elbow_rad    = math.radians(
        float(np.clip(elbow_mapped, *_ELBOW_RANGE))
    )

    return base_rad, shoulder_rad, elbow_rad
