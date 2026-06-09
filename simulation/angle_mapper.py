"""
angle_mapper.py — Converts forearm joystick + wrist roll angles (degrees)
                  to MuJoCo 3-DoF arm joint angles (radians).

Motion type is matched to joint type:

  DoF 1  Wrist Roll    [0–180°, 90=neutral]  ROTATION   → joint_base    [−π/2, π/2]
  DoF 2  Forward Lean  [0–180°, 90=neutral]  INCLINATION→ joint_shoulder [0,   π/2]
  DoF 3  Side Lean     [0–180°, 90=neutral]  INCLINATION→ joint_elbow   [0,   2.27]

Neutral posture (forearm vertical, palm right — all inputs = 90°):
  base_rad     =  0   (arm centred)
  shoulder_rad =  0   (arm pointing straight up)
  elbow_rad    =  0   (arm straight)
"""

from __future__ import annotations

import math
import numpy as np


def human_to_robot(
    wrist_roll_deg:   float,
    forward_lean_deg: float,
    side_lean_deg:    float,
) -> tuple[float, float, float]:
    """
    Map human gestures to MuJoCo joint setpoints (radians).

    Parameters
    ----------
    wrist_roll_deg   : Wrist rotation  [0–180°]  (90° = palm-right / neutral)
    forward_lean_deg : Forearm lean F/B[0–180°]  (90° = vertical   / neutral)
    side_lean_deg    : Forearm lean L/R[0–180°]  (90° = vertical   / neutral)

    Returns
    -------
    (base_rad, shoulder_rad, elbow_rad)
    """
    # ROTATION → joint_base [-π/2, π/2]
    # Palm right (90°) = centred; palm up (180°) = pan right; palm down (0°) = pan left
    base_rad = math.radians(float(np.clip(wrist_roll_deg - 90.0, -90.0, 90.0)))

    # INCLINATION → joint_shoulder [0, π/2]
    # Forearm vertical (90°) = arm up; lean toward camera (→180°) = arm raises horizontal
    shoulder_rad = math.radians(float(np.clip(forward_lean_deg - 90.0, 0.0, 90.0)))

    # INCLINATION → joint_elbow [0, 2.27 rad = 130°]
    # Forearm vertical (90°) = elbow straight; lean right (→180°) = elbow bends fully
    elbow_deg = float(np.clip(side_lean_deg - 90.0, 0.0, 90.0)) / 90.0 * 130.0
    elbow_rad = math.radians(elbow_deg)

    return base_rad, shoulder_rad, elbow_rad
