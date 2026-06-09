"""
angle_mapper.py — Converts forearm joystick + wrist roll angles (degrees)
                  to MuJoCo 3-DoF arm joint angles (radians).

Human gesture → Robot joint:

  DoF 1  Side lean    [0–180°, 90=neutral]  lean L/R  → joint_base    [−π/2,  π/2]
  DoF 2  Forward lean [0–180°, 90=neutral]  lean F/B  → joint_shoulder [0,    π/2]
  DoF 3  Wrist roll   [0–180°, 90=neutral]  twist     → joint_elbow   [0,    2.27]

Neutral posture (forearm vertical, palm right):
  side_lean=90   → base_rad     =  0      (arm centred)
  forward_lean=90→ shoulder_rad =  0      (arm pointing up)
  wrist_roll=90  → elbow_rad    =  0      (arm straight)
"""

from __future__ import annotations

import math
import numpy as np


def human_to_robot(
    side_lean_deg:    float,
    forward_lean_deg: float,
    wrist_roll_deg:   float,
) -> tuple[float, float, float]:
    """
    Map forearm joystick + wrist roll to MuJoCo joint setpoints (radians).

    Parameters
    ----------
    side_lean_deg    : Forearm side lean    [0–180°]  (90° = vertical / centred)
    forward_lean_deg : Forearm forward lean [0–180°]  (90° = vertical / neutral)
    wrist_roll_deg   : Wrist roll           [0–180°]  (90° = palm-right / neutral)

    Returns
    -------
    (base_rad, shoulder_rad, elbow_rad)
    """
    # DoF 1 → joint_base [-π/2, π/2]
    # 90° lean (vertical) = 0 rad (centred); full right (180°) = +π/2; full left (0°) = -π/2
    base_rad = math.radians(float(np.clip(side_lean_deg - 90.0, -90.0, 90.0)))

    # DoF 2 → joint_shoulder [0, π/2]
    # 90° lean (vertical) = 0 rad (arm up); lean forward to 180° = π/2 (arm horizontal)
    # Lean backward (<90°) is clamped to 0 (arm stays up)
    shoulder_rad = math.radians(float(np.clip(forward_lean_deg - 90.0, 0.0, 90.0)))

    # DoF 3 → joint_elbow [0, 2.27 rad = 130°]
    # 90° roll (palm right/neutral) = 0 rad (elbow straight)
    # 180° roll (palm up)           = 130° / 2.27 rad (fully bent)
    # <90° roll (palm down)         = 0 (clamped, elbow stays straight)
    elbow_deg = float(np.clip((wrist_roll_deg - 90.0) / 90.0 * 130.0, 0.0, 130.0))
    elbow_rad = math.radians(elbow_deg)

    return base_rad, shoulder_rad, elbow_rad
