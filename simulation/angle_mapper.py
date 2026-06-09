"""
angle_mapper.py — Converts forearm+hand control angles (degrees) to
                  MuJoCo 3-DoF arm joint angles (radians).

Control posture: forearm vertical, elbow pointing down.

Human gesture → Robot joint mapping:

  DoF 1  Forearm Yaw   [0–180°]  swing wrist L/R  → joint_base   [−π/2, π/2]
  DoF 2  Wrist Tilt    [0–180°]  incline wrist     → joint_shoulder[0,   π/2]
  DoF 3  Wrist Roll    [0–180°]  rotate hand       → joint_elbow  [0,   2.27 rad]

Neutral posture (forearm vertical, wrist straight, palm facing right):
  forearm_yaw ≈ 90°  →  base     = 0 rad  (centred)
  wrist_tilt  ≈  0°  →  shoulder = 0 rad  (arm pointing up)
  wrist_roll  ≈ 90°  →  elbow    = 1.13 rad (half bent)
"""

from __future__ import annotations

import math
import numpy as np


def human_to_robot(
    forearm_yaw_deg: float,
    wrist_tilt_deg:  float,
    wrist_roll_deg:  float,
) -> tuple[float, float, float]:
    """
    Map forearm+hand angles to MuJoCo joint setpoints (radians).

    Parameters
    ----------
    forearm_yaw_deg : Forearm yaw   [0–180°]  (90° = forward / neutral)
    wrist_tilt_deg  : Wrist tilt    [0–180°]  (0°  = straight, 90° = bent 90°)
    wrist_roll_deg  : Wrist roll    [0–180°]  (90° = palm-right / neutral)

    Returns
    -------
    (base_rad, shoulder_rad, elbow_rad) clamped to each joint's safe range.
    """
    # DoF 1 → joint_base [-π/2, π/2]
    # 90° yaw (forward) = 0 rad centred; swing ±90° maps to ±π/2
    base_rad = math.radians(float(np.clip(forearm_yaw_deg - 90.0, -90.0, 90.0)))

    # DoF 2 → joint_shoulder [0, π/2]
    # 0° tilt (wrist straight) = arm up (0 rad); 90° tilt = arm horizontal (π/2)
    shoulder_rad = math.radians(float(np.clip(wrist_tilt_deg, 0.0, 90.0)))

    # DoF 3 → joint_elbow [0, 2.27 rad = 130°]
    # Wrist roll 0°   (palm down) → elbow   0° (straight)
    # Wrist roll 90°  (palm side) → elbow  65° (half bent)
    # Wrist roll 180° (palm up)   → elbow 130° (fully bent)
    elbow_deg = float(np.clip(wrist_roll_deg / 180.0 * 130.0, 0.0, 130.0))
    elbow_rad = math.radians(elbow_deg)

    return base_rad, shoulder_rad, elbow_rad
