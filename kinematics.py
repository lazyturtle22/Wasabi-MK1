"""
kinematics.py - Maps MediaPipe 3D joint coordinates to robot servo angles.

MediaPipe Pose world-coordinate conventions (used throughout this file):
  X  : horizontal — positive = subject's right
  Y  : vertical   — positive = DOWNWARD  (inverted from intuition)
  Z  : depth      — positive = toward camera

All angles are returned as integers in [0, 180] degrees to match
standard servo PWM ranges.

Body-relative frame
-------------------
Base pan and shoulder tilt are computed in the body's LOCAL coordinate
frame (origin = tracked shoulder, axes derived from the shoulder line and
world vertical).  This makes both angles invariant to body translation and
rotation — only the tracked arm's movement relative to the torso matters.

Joint mapping:
  Servo 1 — Base Pan    : horizontal yaw of the upper arm in the body frame
  Servo 2 — Shoulder    : elevation of the upper arm from body-vertical
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


# ── Body-relative frame ────────────────────────────────────────────────────────

def _body_frame(sh_l: np.ndarray, sh_r: np.ndarray):
    """
    Build an orthonormal body frame from the two shoulder positions.

    Returns (body_right, body_up, body_forward) as unit vectors.

    body_right   : left shoulder → right shoulder
    body_up      : perpendicular to body_right, pointing upward (−Y in MediaPipe)
    body_forward : cross(body_right, body_up) — pointing in front of the subject
    """
    body_right = _unit(sh_l, sh_r)               # left → right shoulder

    # MediaPipe +Y is downward, so world-up is (0, -1, 0)
    world_up = np.array([0.0, -1.0, 0.0])

    # Project world_up perpendicular to body_right to get a consistent body_up
    body_up = world_up - np.dot(world_up, body_right) * body_right
    up_norm = float(np.linalg.norm(body_up))
    body_up = body_up / up_norm if up_norm > 1e-7 else world_up

    body_forward = np.cross(body_right, body_up)  # already unit if inputs are

    return body_right, body_up, body_forward


def _body_relative_pan_tilt(shoulder: np.ndarray,
                             sh_l: np.ndarray,
                             sh_r: np.ndarray,
                             elbow:    np.ndarray,
                             use_left_arm: bool) -> tuple[int, int]:
    """
    Compute Base Pan and Shoulder Tilt in the body's local coordinate frame.

    Invariant to body translation, body rotation (yaw/pitch/roll), and
    camera angle — only the arm's pose relative to the torso matters.

    Base Pan
    --------
    Project the upper-arm vector onto the body's horizontal plane
    (spanned by body_right × body_forward) and measure its signed angle
    from body_forward.

    0°   = arm pointing straight out to the side  (away from body centre)
    90°  = arm pointing straight forward           (in front of torso)
    180° = arm crossing the body                  (toward opposite shoulder)

    Shoulder Tilt
    -------------
    Angle between the upper-arm vector and body_up.

    0°   = arm pointing straight up
    90°  = arm horizontal
    180° = arm pointing straight down
    """
    body_right, body_up, body_forward = _body_frame(sh_l, sh_r)
    upper_arm = _unit(shoulder, elbow)

    # ── Base Pan ───────────────────────────────────────────────────────────────
    # Remove the body_up component to project onto the horizontal plane
    proj = upper_arm - np.dot(upper_arm, body_up) * body_up
    proj_norm = float(np.linalg.norm(proj))
    if proj_norm > 1e-7:
        proj = proj / proj_norm
        cos_pan = float(np.clip(np.dot(proj, body_forward), -1.0, 1.0))
        # Signed angle: positive = arm swinging toward body_right
        sin_pan = float(np.dot(np.cross(body_forward, proj), body_up))
        pan_deg = float(np.degrees(np.arctan2(sin_pan, cos_pan)))
    else:
        pan_deg = 0.0

    # For the left arm the pan sign is mirrored so both arms map 90° = forward
    if use_left_arm:
        pan_deg = -pan_deg

    base = _clamp(pan_deg + 90.0)   # shift signed angle → [0, 180]

    # ── Shoulder Tilt ──────────────────────────────────────────────────────────
    tilt = _clamp(_angle_deg(upper_arm, body_up))

    return base, tilt


# ── Individual servo calculations (elbow, wrist, gripper) ─────────────────────

def compute_elbow_tilt(shoulder: np.ndarray, elbow: np.ndarray,
                       wrist: np.ndarray) -> int:
    """
    Elbow Tilt: interior angle at the elbow joint.

    0°   = fully bent (wrist near shoulder)
    180° = arm fully extended / straight
    """
    to_shoulder = _unit(elbow, shoulder)
    to_wrist    = _unit(elbow, wrist)
    return _clamp(_angle_deg(to_shoulder, to_wrist))


def compute_wrist_pitch(elbow: np.ndarray, wrist: np.ndarray,
                        index_mcp: np.ndarray) -> int:
    """
    Wrist Pitch: flexion / extension of the hand relative to the forearm.

    0°   = hand maximally flexed
    90°  = wrist neutral / straight
    180° = hand maximally extended
    """
    forearm_dir = _unit(wrist, elbow)
    hand_dir    = _unit(wrist, index_mcp)
    return _clamp(_angle_deg(forearm_dir, hand_dir))


def compute_wrist_roll(wrist: np.ndarray, index_mcp: np.ndarray,
                       pinky_mcp: np.ndarray) -> int:
    """
    Wrist Roll (pronation / supination).

    0°   = palm facing down
    90°  = palm facing sideways
    180° = palm facing up
    """
    knuckle_span = _unit(index_mcp, pinky_mcp)
    elevation = float(np.degrees(np.arcsin(np.clip(-knuckle_span[1], -1.0, 1.0))))
    return _clamp(elevation + 90.0)


def compute_gripper(thumb_tip: np.ndarray, index_tip: np.ndarray,
                    open_ref: float = 0.10) -> int:
    """
    Gripper angle from thumb-tip to index-tip distance.

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
    pose_landmarks : landmark list from MediaPipe Pose (via _LandmarkList wrapper)
    hand_landmarks : landmark list from MediaPipe Hands, or None
    use_left_arm   : True  → track subject's LEFT arm
                     False → track subject's RIGHT arm  (recommended for control)

    Returns
    -------
    [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper] or None if
    pose landmarks are absent.

    MediaPipe Pose landmark indices used:
        11 = LEFT_SHOULDER   12 = RIGHT_SHOULDER
        13 = LEFT_ELBOW      14 = RIGHT_ELBOW
        15 = LEFT_WRIST      16 = RIGHT_WRIST
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark

    if use_left_arm:
        s_idx, e_idx, w_idx = 11, 13, 15
    else:
        s_idx, e_idx, w_idx = 12, 14, 16

    shoulder   = _lm_to_np(lm[s_idx])
    elbow      = _lm_to_np(lm[e_idx])
    wrist_pose = _lm_to_np(lm[w_idx])
    sh_l       = _lm_to_np(lm[11])
    sh_r       = _lm_to_np(lm[12])

    # Body-relative pan and tilt — unaffected by body translation or rotation
    base, shoulder_angle = _body_relative_pan_tilt(
        shoulder, sh_l, sh_r, elbow, use_left_arm
    )
    elbow_angle = compute_elbow_tilt(shoulder, elbow, wrist_pose)

    # Default wrist / gripper values during hand-detection dropout
    wrist_pitch = 90
    wrist_roll  = 90
    gripper     = 45

    if hand_landmarks is not None:
        hl = hand_landmarks.landmark
        wrist_h   = _lm_to_np(hl[0])
        index_mcp = _lm_to_np(hl[5])
        pinky_mcp = _lm_to_np(hl[17])
        thumb_tip = _lm_to_np(hl[4])
        index_tip = _lm_to_np(hl[8])

        wrist_pitch = compute_wrist_pitch(wrist_pose, wrist_h, index_mcp)
        wrist_roll  = compute_wrist_roll(wrist_h, index_mcp, pinky_mcp)
        gripper     = compute_gripper(thumb_tip, index_tip)

    return [base, shoulder_angle, elbow_angle, wrist_pitch, wrist_roll, gripper]
