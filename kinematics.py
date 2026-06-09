"""
kinematics.py - Maps MediaPipe forearm+hand pose to 3-DoF robot servo angles.

Control posture: hold forearm roughly vertical with elbow pointing DOWN and
wrist pointing UP.  Three gestures drive the robot:

  DoF 1 — Forearm Yaw   : swing the wrist left/right around the elbow pivot
                           (rotation around the vertical / elbow axis)
  DoF 2 — Wrist Tilt    : incline/bend the wrist relative to the forearm axis
                           (flexion / extension of the wrist joint)
  DoF 3 — Wrist Roll    : rotate the hand around the forearm axis
                           (pronation / supination)

Required landmarks
  Pose  : elbow (13 or 14), wrist (15 or 16)
  Hands : wrist(0), index_mcp(5), pinky_mcp(17)

MediaPipe world-coordinate conventions:
  X  positive = subject's right
  Y  positive = DOWNWARD  (inverted)
  Z  positive = toward camera

All angles returned as integers in [0, 180]° for servo compatibility.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional


# ── Geometry primitives ────────────────────────────────────────────────────────

def _unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector from *a* to *b*; returns zero vector if degenerate."""
    v = b - a
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-7 else np.zeros(3, dtype=float)


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-7 or n2 < 1e-7:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _clamp(value: float, lo: float = 0.0, hi: float = 180.0) -> int:
    return int(np.clip(round(value), lo, hi))


def _lm_to_np(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def _project_perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Remove the component of *v* along *axis* (axis must be a unit vector)."""
    return v - np.dot(v, axis) * axis


# ── 3-DoF forearm+hand calculations ───────────────────────────────────────────

def compute_forearm_yaw(elbow: np.ndarray, wrist: np.ndarray) -> int:
    """
    DoF 1 — Forearm Yaw.

    Project the elbow→wrist vector onto the horizontal plane and measure
    its compass bearing relative to the camera's forward (−Z) axis.

    Hold forearm vertical.  Swing wrist left/right to pan the robot base.

    0°   = forearm points to the subject's right
    90°  = forearm points toward camera  (forward / neutral)
    180° = forearm points to the subject's left
    """
    fw = wrist - elbow
    horiz = np.array([fw[0], 0.0, fw[2]], dtype=float)
    horiz_norm = float(np.linalg.norm(horiz))
    if horiz_norm < 1e-7:
        return 90   # forearm is perfectly vertical — hold neutral
    angle = float(np.degrees(np.arctan2(horiz[0], -horiz[2])))
    return _clamp(angle + 90.0)


def compute_wrist_tilt(elbow: np.ndarray, wrist: np.ndarray,
                       wrist_h: np.ndarray, index_mcp: np.ndarray) -> int:
    """
    DoF 2 — Wrist Tilt (inclination / flexion-extension).

    Angle between the forearm axis (elbow→wrist) and the hand axis
    (hand-wrist→index-MCP).

    0°   = wrist straight (hand continues forearm direction)
    90°  = wrist bent 90°
    180° = wrist bent fully back (uncommon in practice)

    Incline/extend wrist forward to tilt the robot shoulder up.
    """
    forearm_dir = _unit(elbow, wrist)
    hand_dir    = _unit(wrist_h, index_mcp)
    return _clamp(_angle_deg(forearm_dir, hand_dir))


def compute_wrist_roll(elbow: np.ndarray, wrist: np.ndarray,
                       index_mcp: np.ndarray, pinky_mcp: np.ndarray) -> int:
    """
    DoF 3 — Wrist Roll (rotation around the forearm axis).

    Measures the angle of the knuckle-span vector (index→pinky) projected
    onto the plane perpendicular to the forearm, relative to world-right (+X).

    Hold palm facing right → ~90° (neutral).
    Rotate hand so palm faces up → ~180°.
    Rotate hand so palm faces down → ~0°.

    Roll the wrist to drive the robot elbow joint.
    """
    forearm_dir = _unit(elbow, wrist)

    # Knuckle span projected perpendicular to forearm axis
    knuckle_raw  = index_mcp - pinky_mcp
    knuckle_perp = _project_perp(knuckle_raw, forearm_dir)
    kp_norm      = float(np.linalg.norm(knuckle_perp))
    if kp_norm < 1e-7:
        return 90

    knuckle_perp = knuckle_perp / kp_norm

    # Reference vector: world-right (+X) projected perp to forearm.
    # Works well when the forearm is vertical (forearm ≈ −Y, so +X is
    # already perpendicular).
    world_right = np.array([1.0, 0.0, 0.0])
    ref = _project_perp(world_right, forearm_dir)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-7:
        return 90

    ref = ref / ref_norm

    cos_r = float(np.clip(np.dot(knuckle_perp, ref), -1.0, 1.0))
    sin_r = float(np.dot(np.cross(ref, knuckle_perp), forearm_dir))
    roll  = float(np.degrees(np.arctan2(sin_r, cos_r)))
    return _clamp(roll + 90.0)


# ── Master conversion function ─────────────────────────────────────────────────

def joints_to_angles(pose_landmarks: Any,
                     hand_landmarks: Any,
                     use_left_arm: bool = False) -> Optional[list[int]]:
    """
    Convert forearm + hand pose into three robot servo angles.

    Parameters
    ----------
    pose_landmarks : landmark list from MediaPipe Pose (via _LandmarkList)
    hand_landmarks : landmark list from MediaPipe Hands, or None
    use_left_arm   : False = right arm/hand (default for this control scheme)
                     True  = left arm/hand

    Returns
    -------
    [forearm_yaw, wrist_tilt, wrist_roll, 90, 90, 45]
    First three entries drive the 3-DoF robot; last three are unused placeholders.
    Returns None if pose landmarks are absent.

    Pose landmark indices:
        13 = LEFT_ELBOW    14 = RIGHT_ELBOW
        15 = LEFT_WRIST    16 = RIGHT_WRIST
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark

    e_idx, w_idx = (13, 15) if use_left_arm else (14, 16)

    elbow = _lm_to_np(lm[e_idx])
    wrist = _lm_to_np(lm[w_idx])

    # DoF 1: always computable from pose alone
    forearm_yaw = compute_forearm_yaw(elbow, wrist)

    # DoF 2 & 3: need hand landmarks
    wrist_tilt = 90
    wrist_roll = 90

    if hand_landmarks is not None:
        hl = hand_landmarks.landmark

        wrist_h   = _lm_to_np(hl[0])    # WRIST
        index_mcp = _lm_to_np(hl[5])    # INDEX_FINGER_MCP
        pinky_mcp = _lm_to_np(hl[17])   # PINKY_MCP

        wrist_tilt = compute_wrist_tilt(elbow, wrist, wrist_h, index_mcp)
        wrist_roll = compute_wrist_roll(elbow, wrist, index_mcp, pinky_mcp)

    return [forearm_yaw, wrist_tilt, wrist_roll, 90, 90, 45]
