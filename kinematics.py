"""
kinematics.py - Maps MediaPipe forearm+hand pose to 3-DoF robot servo angles.

Control posture
---------------
Hold the forearm roughly vertical — elbow pointing DOWN, wrist pointing UP.
The forearm acts as a 2-axis joystick; the wrist adds a third twist axis.

  DoF 1 — Side Lean    : tilt forearm left / right  (rotation around depth axis)
  DoF 2 — Forward Lean : tilt forearm toward / away  (rotation around side axis)
  DoF 3 — Wrist Roll   : rotate hand around forearm  (pronation / supination)

Why these three work
---------------------
Each DoF is a rotation around one clear anatomical axis, measured as an
angle of the forearm or knuckle-span unit vector from a fixed reference.
The maths is identical in structure to the working wrist-roll calculation.

  DoF 1 angle = atan2(forearm.x,  −forearm.y)   in the frontal   (XY) plane
  DoF 2 angle = atan2(forearm.z,  −forearm.y)   in the sagittal  (ZY) plane
  DoF 3 angle = atan2(sin, cos)  of knuckle-span projected ⊥ forearm

DoF 1 & 2 need only the elbow + wrist pose landmarks.
DoF 3 needs the hand landmarks (index_mcp, pinky_mcp).

Neutral posture (all outputs = 90°, robot centred / arm up / elbow half-bent):
  Forearm perfectly vertical, palm facing to the right.

MediaPipe world-coordinate conventions:
  X  positive = subject's right
  Y  positive = DOWNWARD  (inverted)
  Z  positive = toward camera
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Optional


# ── Geometry primitives ────────────────────────────────────────────────────────

def _unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-7 else np.zeros(3, dtype=float)


def _clamp(value: float, lo: float = 0.0, hi: float = 180.0) -> int:
    return int(np.clip(round(value), lo, hi))


def _lm_to_np(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def _project_perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Remove the component of *v* along *axis* (axis assumed unit length)."""
    return v - np.dot(v, axis) * axis


# ── Three DoF calculations ─────────────────────────────────────────────────────

def compute_side_lean(elbow: np.ndarray, wrist: np.ndarray) -> int:
    """
    DoF 1 — Forearm side lean (left / right tilt from vertical).

    Measured as the angle of the forearm unit vector in the frontal (X-Y)
    plane, relative to the upward vertical (−Y in MediaPipe).

    90°  = forearm perfectly vertical  (neutral / centred)
    >90° = forearm leaning to the RIGHT
    <90° = forearm leaning to the LEFT

    → Controls robot base pan.
    """
    fw = _unit(elbow, wrist)
    # atan2(x_component, upward_component):  upward = −Y in MediaPipe
    angle = math.degrees(math.atan2(fw[0], -fw[1]))
    return _clamp(angle + 90.0)


def compute_forward_lean(elbow: np.ndarray, wrist: np.ndarray) -> int:
    """
    DoF 2 — Forearm forward lean (toward / away from camera).

    Measured as the angle of the forearm unit vector in the sagittal (Z-Y)
    plane, relative to the upward vertical (−Y in MediaPipe).

    90°  = forearm perfectly vertical  (neutral)
    >90° = forearm leaning toward the camera  (forward)
    <90° = forearm leaning away from the camera (backward)

    → Controls robot shoulder elevation.
    """
    fw = _unit(elbow, wrist)
    # atan2(z_component, upward_component)
    angle = math.degrees(math.atan2(fw[2], -fw[1]))
    return _clamp(angle + 90.0)


def compute_wrist_roll(elbow: np.ndarray, wrist: np.ndarray,
                       index_mcp: np.ndarray, pinky_mcp: np.ndarray) -> int:
    """
    DoF 3 — Wrist roll (rotation of hand around the forearm axis).

    Projects the knuckle-span vector (index→pinky) perpendicular to the
    forearm axis and measures its angle relative to world-right (+X),
    using the forearm axis itself as the rotation reference.

    90°  = palm facing to the right  (neutral)
    180° = palm facing up            (supinated)
    0°   = palm facing down          (pronated)

    → Controls robot elbow bend.
    """
    forearm_dir  = _unit(elbow, wrist)

    knuckle_raw  = index_mcp - pinky_mcp
    knuckle_perp = _project_perp(knuckle_raw, forearm_dir)
    kp_norm      = float(np.linalg.norm(knuckle_perp))
    if kp_norm < 1e-7:
        return 90

    knuckle_perp /= kp_norm

    # Reference: world-right (+X) projected perpendicular to forearm.
    # When the forearm is vertical this is identical to +X — stable and clear.
    world_right = np.array([1.0, 0.0, 0.0])
    ref = _project_perp(world_right, forearm_dir)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-7:
        return 90

    ref /= ref_norm

    cos_r = float(np.clip(np.dot(knuckle_perp, ref), -1.0, 1.0))
    sin_r = float(np.dot(np.cross(ref, knuckle_perp), forearm_dir))
    return _clamp(math.degrees(math.atan2(sin_r, cos_r)) + 90.0)


# ── Master conversion function ─────────────────────────────────────────────────

def joints_to_angles(pose_landmarks: Any,
                     hand_landmarks: Any,
                     use_left_arm: bool = False) -> Optional[list[int]]:
    """
    Convert forearm + hand pose into three robot servo angles.

    Parameters
    ----------
    pose_landmarks : landmark list from MediaPipe Pose
    hand_landmarks : landmark list from MediaPipe Hands, or None
    use_left_arm   : False = right arm (default); True = left arm

    Returns
    -------
    [side_lean, forward_lean, wrist_roll, 90, 90, 45]
    Indices 0-2 drive the 3-DoF robot; 3-5 are unused placeholders.
    Returns None if pose landmarks are absent.

    Pose landmark indices used:
        13 = LEFT_ELBOW    14 = RIGHT_ELBOW
        15 = LEFT_WRIST    16 = RIGHT_WRIST
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark
    e_idx, w_idx = (13, 15) if use_left_arm else (14, 16)

    elbow = _lm_to_np(lm[e_idx])
    wrist = _lm_to_np(lm[w_idx])

    side_lean    = compute_side_lean(elbow, wrist)
    forward_lean = compute_forward_lean(elbow, wrist)

    # Wrist roll requires hand landmarks; fall back to neutral if not detected
    wrist_roll = 90
    if hand_landmarks is not None:
        hl = hand_landmarks.landmark
        index_mcp = _lm_to_np(hl[5])
        pinky_mcp = _lm_to_np(hl[17])
        wrist_roll = compute_wrist_roll(elbow, wrist, index_mcp, pinky_mcp)

    return [side_lean, forward_lean, wrist_roll, 90, 90, 45]
