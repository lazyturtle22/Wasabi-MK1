"""
kinematics.py - Maps MediaPipe 3D joint coordinates to robot servo angles.

MediaPipe Pose world-coordinate conventions (used throughout this file):
  X  : horizontal — positive = subject's right
  Y  : vertical   — positive = DOWNWARD  (inverted from intuition)
  Z  : depth      — positive = toward camera

All angles are returned as integers in [0, 180] degrees to match
standard servo PWM ranges.

Joint mapping:
  Servo 1 — Base Pan    : yaw  of the upper arm in the horizontal plane
  Servo 2 — Shoulder    : elevation angle of the upper arm from vertical
  Servo 3 — Elbow       : interior bend angle at the elbow
  Servo 4 — Wrist Pitch : hand flexion / extension relative to forearm
  Servo 5 — Wrist Roll  : pronation / supination of the hand
  Servo 6 — Gripper     : thumb-tip to index-tip pinch distance
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional


# ── Geometry primitives ────────────────────────────────────────────────────────

def _unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the unit vector pointing from *a* to *b*."""
    v = b - a
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-7 else np.zeros(3, dtype=float)


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors (need not be unit vectors)."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-7 or n2 < 1e-7:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _clamp(value: float, lo: float = 0.0, hi: float = 180.0) -> int:
    return int(np.clip(round(value), lo, hi))


def _lm_to_np(landmark) -> np.ndarray:
    """Convert a MediaPipe landmark object to a numpy (x, y, z) array."""
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


# ── Individual servo calculations ──────────────────────────────────────────────

def compute_base_pan(shoulder_l: np.ndarray, shoulder_r: np.ndarray,
                     elbow: np.ndarray) -> int:
    """
    Base Pan (yaw): horizontal rotation of the arm relative to the torso.

    Strategy: project the mid-shoulder → elbow vector onto the XZ plane
    (ignoring vertical component) and measure its signed angle versus the
    camera's forward axis (-Z).

    0°  = arm sweeping fully to the robot's right
    90° = arm pointing straight forward (toward camera)
    180°= arm sweeping fully to the robot's left
    """
    mid = (shoulder_l + shoulder_r) / 2.0
    xz  = np.array([elbow[0] - mid[0], 0.0, elbow[2] - mid[2]], dtype=float)
    if np.linalg.norm(xz) < 1e-7:
        return 90
    angle = float(np.degrees(np.arctan2(xz[0], -xz[2])))   # signed [-180, 180]
    return _clamp(angle + 90.0)                              # shift to [0, 180]


def compute_shoulder_tilt(shoulder: np.ndarray, elbow: np.ndarray) -> int:
    """
    Shoulder Tilt: elevation of the upper arm from the downward vertical.

    0°   = arm pointing straight up   (fully raised)
    90°  = arm horizontal
    180° = arm pointing straight down (hanging relaxed)
    """
    upper_arm = _unit(shoulder, elbow)
    downward  = np.array([0.0, 1.0, 0.0])   # +Y is downward in MediaPipe
    return _clamp(_angle_deg(upper_arm, downward))


def compute_elbow_tilt(shoulder: np.ndarray, elbow: np.ndarray,
                       wrist: np.ndarray) -> int:
    """
    Elbow Tilt: interior angle at the elbow joint.

    0°   = fully bent (wrist near shoulder)
    180° = arm fully extended / straight
    """
    # Both vectors originate at the elbow
    to_shoulder = _unit(elbow, shoulder)
    to_wrist    = _unit(elbow, wrist)
    return _clamp(_angle_deg(to_shoulder, to_wrist))


def compute_wrist_pitch(elbow: np.ndarray, wrist: np.ndarray,
                        index_mcp: np.ndarray) -> int:
    """
    Wrist Pitch: flexion / extension of the hand relative to the forearm.

    Uses the wrist → index-MCP vector as the 'hand axis'.
    0°   = hand maximally flexed (palm toward forearm)
    90°  = wrist neutral / straight
    180° = hand maximally extended (back of hand toward forearm)
    """
    forearm_dir = _unit(wrist, elbow)       # pointing back up the forearm
    hand_dir    = _unit(wrist, index_mcp)   # pointing into the hand
    return _clamp(_angle_deg(forearm_dir, hand_dir))


def compute_wrist_roll(wrist: np.ndarray, index_mcp: np.ndarray,
                       pinky_mcp: np.ndarray) -> int:
    """
    Wrist Roll (pronation / supination): twist of the hand around the forearm.

    Computes the elevation angle of the knuckle-span vector (index → pinky).
    When the palm is flat and horizontal the knuckle span is horizontal → 0°
    elevation, mapped to 90° servo output.

    0°   = hand fully pronated  (palm facing down)
    90°  = palm facing sideways / neutral
    180° = hand fully supinated (palm facing up)
    """
    knuckle_span = _unit(index_mcp, pinky_mcp)
    # arcsin of the Y component gives signed elevation; Y is inverted in MediaPipe
    elevation = float(np.degrees(np.arcsin(np.clip(-knuckle_span[1], -1.0, 1.0))))
    return _clamp(elevation + 90.0)   # shift [-90, 90] → [0, 180]


def compute_gripper(thumb_tip: np.ndarray, index_tip: np.ndarray,
                    open_ref: float = 0.10) -> int:
    """
    Gripper angle from the Euclidean distance between thumb tip and index tip.

    'open_ref' is the approximate distance (in MediaPipe world units, ~metres)
    that corresponds to a fully open hand. Tune this once the camera is set up.

    0°  = fingers pinched closed
    90° = hand wide open
    """
    dist  = float(np.linalg.norm(thumb_tip - index_tip))
    ratio = np.clip(dist / open_ref, 0.0, 1.0)
    return _clamp(ratio * 90.0, lo=0.0, hi=90.0)


# ── Master conversion function ─────────────────────────────────────────────────

def joints_to_angles(pose_landmarks: Any,
                     hand_landmarks: Any,
                     use_left_arm: bool = True) -> Optional[list[int]]:
    """
    Convert a frame's MediaPipe landmark objects into six servo angles.

    Parameters
    ----------
    pose_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Full-body pose landmarks from MediaPipe Pose.
    hand_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList | None
        Single-hand landmarks from MediaPipe Hands, or None if not detected.
    use_left_arm : bool
        True  → track the subject's LEFT arm  (operator mirrors own left arm → robot)
        False → track the subject's RIGHT arm

    Returns
    -------
    list[int] | None
        [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]  or None if
        pose landmarks are absent.

    MediaPipe Pose landmark indices (subset used here):
        11 = LEFT_SHOULDER   12 = RIGHT_SHOULDER
        13 = LEFT_ELBOW      14 = RIGHT_ELBOW
        15 = LEFT_WRIST      16 = RIGHT_WRIST
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark

    # Select arm side
    if use_left_arm:
        s_idx, e_idx, w_idx = 11, 13, 15
        opp_shoulder_idx    = 12
    else:
        s_idx, e_idx, w_idx = 12, 14, 16
        opp_shoulder_idx    = 11

    shoulder   = _lm_to_np(lm[s_idx])
    elbow      = _lm_to_np(lm[e_idx])
    wrist_pose = _lm_to_np(lm[w_idx])
    sh_l       = _lm_to_np(lm[11])
    sh_r       = _lm_to_np(lm[12])

    base     = compute_base_pan(sh_l, sh_r, elbow)
    shoulder_angle = compute_shoulder_tilt(shoulder, elbow)
    elbow_angle    = compute_elbow_tilt(shoulder, elbow, wrist_pose)

    # Default wrist / gripper values used when hand detector has no lock
    wrist_pitch = 90
    wrist_roll  = 90
    gripper     = 45

    if hand_landmarks is not None:
        hl = hand_landmarks.landmark

        # MediaPipe Hands landmark indices:
        #   0 = WRIST
        #   5 = INDEX_FINGER_MCP   17 = PINKY_MCP
        #   4 = THUMB_TIP          8  = INDEX_FINGER_TIP
        wrist_h   = _lm_to_np(hl[0])
        index_mcp = _lm_to_np(hl[5])
        pinky_mcp = _lm_to_np(hl[17])
        thumb_tip = _lm_to_np(hl[4])
        index_tip = _lm_to_np(hl[8])

        wrist_pitch = compute_wrist_pitch(wrist_pose, wrist_h, index_mcp)
        wrist_roll  = compute_wrist_roll(wrist_h, index_mcp, pinky_mcp)
        gripper     = compute_gripper(thumb_tip, index_tip)

    return [base, shoulder_angle, elbow_angle, wrist_pitch, wrist_roll, gripper]
