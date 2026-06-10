"""
retarget.py — Maps MediaPipe 3-D landmarks onto the Wasabi-MK1 arm (RIGHT arm only).

Why this exists (and why kinematics.py was replaced)
----------------------------------------------------
The old mapping computed each servo angle independently in *camera* coordinates,
so the output changed whenever the user leaned, turned, or stood off-axis, and
the base pan depended entirely on MediaPipe's monocular depth (its noisiest
estimate). This module instead:

  1. Gates every landmark on its visibility score.
  2. One-Euro-filters the raw 3-D landmark positions (filtering positions is
     strictly better than filtering output angles — no wrap-around artifacts).
  3. Builds a TORSO-ANCHORED reference frame from the shoulders + hips, with
     the anterior direction disambiguated by the nose. All arm vectors are
     expressed in this frame, making the output invariant to camera placement
     and body orientation.
  4. Default "ik" mode: the wrist position relative to the right shoulder is
     scaled by an auto-calibrated arm-length ratio and fed to an analytic
     3-DoF inverse-kinematics solver (base yaw + planar 2R). The simulated
     end-effector therefore tracks the *position of your hand* exactly —
     which is what precision teleoperation needs.
     "joint" mode: direct per-joint angle extraction (pose mimicry).
  5. Wrist roll is measured as a signed rotation about the forearm axis
     (camera-invariant), gripper from normalized thumb-index pinch.
  6. Yaw is held near the vertical singularity (arm hanging ~straight down,
     where azimuth is mathematically undefined), all joints are clamped to the
     simulation's ranges, and a slew-rate limiter caps command velocity.

Robot frame convention (matches simulation/arm.xml):
  x = robot forward, y = robot left, z = up. Shoulder joint at the origin.
  q_base     : yaw about z. 0 = forward, positive toward robot's left.
  q_shoulder : upper-arm pitch above horizontal. 0 = horizontal, +90 = up.
  q_elbow    : flexion from straight. 0 = arm straight, positive = curl.
  q_wrist    : roll about the forearm axis. 0 = neutral, positive = supination.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from filters import OneEuroFilter, SlewRateLimiter


# ── MediaPipe landmark indices ─────────────────────────────────────────────────

NOSE, L_SH, R_SH = 0, 11, 12
L_EL, R_EL, L_WR, R_WR = 13, 14, 15, 16
L_HIP, R_HIP = 23, 24

H_WRIST, H_THUMB_TIP, H_INDEX_MCP = 0, 4, 5
H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP = 8, 9, 17

_POSE_SUBSET = (NOSE, L_SH, R_SH, R_EL, R_WR, L_HIP, R_HIP)
_HAND_SUBSET = (H_WRIST, H_THUMB_TIP, H_INDEX_MCP,
                H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class RetargetConfig:
    mode: str = "ik"               # "ik" (end-effector tracking) | "joint" (pose mimicry)
    mirror: bool = True            # True: robot acts like your mirror image (intuitive)

    # Robot geometry — MUST match simulation/arm.xml
    l1: float = 0.30               # upper-arm link length (m)
    l2: float = 0.25               # forearm link length (m)

    # Joint limits (radians) — MUST match simulation/arm.xml
    base_range: tuple = (-2.094, 2.094)        # ±120°
    shoulder_range: tuple = (-1.658, 1.658)    # ±95°
    elbow_range: tuple = (0.0, 2.618)          # 0–150°
    wrist_range: tuple = (-3.1416, 3.1416)     # ±180°

    # Landmark filtering (One Euro)
    min_cutoff: float = 1.0
    beta: float = 0.6
    d_cutoff: float = 1.0

    # Gating / robustness
    visibility_min: float = 0.5    # pose landmarks below this are untrusted
    yaw_lock_lo: float = 0.10      # horiz. reach fraction below which yaw is frozen
    yaw_lock_hi: float = 0.25      # ...and above which yaw fully tracks
    dropout_reset_s: float = 0.7   # tracking gap that resets the filters

    # Output conditioning
    max_joint_speed: float = 12.0  # rad/s slew limit on joint commands
    base_sign: float = 1.0         # flip if base pan feels inverted
    roll_sign: float = 1.0         # flip if wrist roll feels inverted

    # Gripper pinch normalization (pinch dist / palm length)
    pinch_closed: float = 0.25
    pinch_open: float = 1.10


@dataclass
class JointTargets:
    """One frame of robot commands. Angles in radians, grip in [0, 1] (1 = open)."""
    base: float = 0.0
    shoulder: float = 0.6
    elbow: float = 0.9
    wrist_roll: float = 0.0
    grip: float = 1.0
    tracking: bool = False                    # pose lock this frame?
    hand_tracking: bool = False               # hand lock this frame?
    target_xyz: Optional[np.ndarray] = None   # IK target in robot frame (for marker)
    scale: float = 1.0                        # human→robot calibration scale
    debug: dict = field(default_factory=dict)

    def as_degrees(self) -> list[float]:
        return [math.degrees(self.base), math.degrees(self.shoulder),
                math.degrees(self.elbow), math.degrees(self.wrist_roll),
                self.grip * 100.0]


# ── Small math helpers ─────────────────────────────────────────────────────────

def _v(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def _vis(lm) -> float:
    return float(getattr(lm, "visibility", 1.0))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def _ang_diff(a: float, b: float) -> float:
    """Shortest signed angular distance from a to b, in (-pi, pi]."""
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# ── Torso reference frame ──────────────────────────────────────────────────────

def torso_frame(pts: dict[int, np.ndarray],
                hips_ok: bool,
                nose_ok: bool) -> Optional[np.ndarray]:
    """
    Build a 3x3 matrix whose ROWS are the subject's (right, anterior, up) unit
    axes, derived purely from the landmarks themselves — no assumptions about
    MediaPipe's world-axis conventions. `M @ v` converts a world vector into
    (rightward, forward, upward) components.

    right    : left-shoulder → right-shoulder
    up       : mid-hip → mid-shoulder (falls back to camera vertical if the
               hips are out of frame, e.g. sitting at a desk)
    anterior : perpendicular to both, sign disambiguated by the nose, which
               always sits anterior to the shoulder line.
    """
    r = _unit(pts[R_SH] - pts[L_SH])
    if np.allclose(r, 0):
        return None

    mid_sh = (pts[R_SH] + pts[L_SH]) / 2.0
    if hips_ok:
        up_raw = _unit(mid_sh - (pts[R_HIP] + pts[L_HIP]) / 2.0)
    else:
        # Camera vertical: MediaPipe world y points downward in image space.
        up_raw = np.array([0.0, -1.0, 0.0])

    u = _unit(up_raw - np.dot(up_raw, r) * r)
    if np.allclose(u, 0):
        return None

    f = np.cross(r, u)
    # The nose is anterior to the shoulder line — use it to orient `f`.
    if nose_ok:
        ref = pts[NOSE] - mid_sh
        ref_perp = ref - np.dot(ref, u) * u
        if np.linalg.norm(ref_perp) > 1e-6 and np.dot(f, ref_perp) < 0:
            f = -f
    else:
        # Assume the subject faces the camera (world -z is toward the camera).
        if f[2] > 0:
            f = -f

    return np.vstack([r, f, u])


# ── Analytic kinematics (yaw + planar 2R) ──────────────────────────────────────

def fk(q1: float, q2: float, q3: float, l1: float, l2: float) -> np.ndarray:
    """Forward kinematics: joint angles → end-effector xyz in the robot frame."""
    h = l1 * math.cos(q2) + l2 * math.cos(q2 + q3)
    z = l1 * math.sin(q2) + l2 * math.sin(q2 + q3)
    return np.array([h * math.cos(q1), h * math.sin(q1), z])


def solve_ik(target: np.ndarray, l1: float, l2: float,
             prev_yaw: float, lock_lo: float, lock_hi: float
             ) -> tuple[float, float, float, np.ndarray]:
    """
    Analytic IK for the yaw + 2R arm, elbow-down branch (anatomically the only
    branch a human elbow can produce, so retargeted motion stays natural).

    Near the vertical singularity (target almost straight above/below the base,
    where yaw is undefined and numerically explosive) the yaw blends toward
    holding its previous value instead of spinning wildly.

    Returns (q1, q2, q3, clamped_target).
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])
    reach = l1 + l2

    # Clamp target into the reachable annulus, preserving direction.
    d = math.sqrt(x * x + y * y + z * z)
    d_min, d_max = abs(l1 - l2) + 1e-4, reach - 1e-4
    if d < 1e-9:
        x, d = d_min, d_min
    elif not (d_min <= d <= d_max):
        s = min(max(d, d_min), d_max) / d
        x, y, z, d = x * s, y * s, z * s, min(max(d, d_min), d_max)

    # Yaw with singularity hold.
    h = math.hypot(x, y)
    raw_yaw = math.atan2(y, x) if h > 1e-9 else prev_yaw
    w = _smoothstep((h / reach - lock_lo) / max(lock_hi - lock_lo, 1e-6))
    q1 = prev_yaw + w * _ang_diff(prev_yaw, raw_yaw)

    # Planar 2R, elbow-down: q3 = flexion from straight, always >= 0.
    cos_q3 = (d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    q3 = math.acos(min(max(cos_q3, -1.0), 1.0))
    q2 = math.atan2(z, h) - math.atan2(l2 * math.sin(q3),
                                       l1 + l2 * math.cos(q3))
    return q1, q2, q3, np.array([x, y, z])


# ── Per-joint extraction (joint mode) ──────────────────────────────────────────

def joint_angles_direct(upper_rob: np.ndarray, s_world: np.ndarray,
                        e_world: np.ndarray, w_world: np.ndarray,
                        prev_yaw: float, lock_lo: float, lock_hi: float
                        ) -> tuple[float, float, float]:
    """Pose-mimicry mapping: copy the human's own joint angles onto the robot."""
    h = math.hypot(upper_rob[0], upper_rob[1])
    n = float(np.linalg.norm(upper_rob))
    raw_yaw = math.atan2(upper_rob[1], upper_rob[0]) if h > 1e-9 else prev_yaw
    w = _smoothstep((h / max(n, 1e-9) - lock_lo) / max(lock_hi - lock_lo, 1e-6))
    q1 = prev_yaw + w * _ang_diff(prev_yaw, raw_yaw)

    q2 = math.atan2(upper_rob[2], h)

    # Elbow flexion is frame-independent: pi minus the interior elbow angle.
    a = _unit(s_world - e_world)
    b = _unit(w_world - e_world)
    interior = math.acos(min(max(float(np.dot(a, b)), -1.0), 1.0))
    q3 = math.pi - interior
    return q1, q2, q3


# ── Wrist roll & grip ──────────────────────────────────────────────────────────

def wrist_roll(forearm_axis: np.ndarray, knuckle_pinky_to_index: np.ndarray,
               torso_up: np.ndarray) -> Optional[float]:
    """
    Signed pronation/supination about the forearm axis.

    Reference = torso-up projected perpendicular to the forearm; measured
    vector = pinky→index knuckle span, same projection. With the right hand in
    a neutral handshake pose (index knuckle on top) the roll is 0; palm-down
    (pronated) is negative, palm-up (supinated) positive.
    """
    a = _unit(forearm_axis)
    ref = torso_up - np.dot(torso_up, a) * a
    k = knuckle_pinky_to_index - np.dot(knuckle_pinky_to_index, a) * a
    if np.linalg.norm(ref) < 1e-6 or np.linalg.norm(k) < 1e-6:
        return None
    ref, k = _unit(ref), _unit(k)
    return math.atan2(float(np.dot(np.cross(ref, k), a)), float(np.dot(ref, k)))


def grip_fraction(thumb_tip: np.ndarray, index_tip: np.ndarray,
                  wrist: np.ndarray, middle_mcp: np.ndarray,
                  closed: float, open_: float) -> Optional[float]:
    """Pinch distance normalized by palm length → [0 closed, 1 open]."""
    palm = float(np.linalg.norm(middle_mcp - wrist))
    if palm < 1e-6:
        return None
    ratio = float(np.linalg.norm(thumb_tip - index_tip)) / palm
    return min(max((ratio - closed) / (open_ - closed), 0.0), 1.0)


# ── Main retargeter ────────────────────────────────────────────────────────────

class ArmRetargeter:
    """Stateful frame-to-frame retargeter. Call update() once per camera frame."""

    def __init__(self, config: RetargetConfig | None = None):
        self.cfg = config or RetargetConfig()
        c = self.cfg
        self._pose_filter = OneEuroFilter(c.min_cutoff, c.beta, c.d_cutoff)
        self._hand_filter = OneEuroFilter(c.min_cutoff, c.beta, c.d_cutoff)
        self._slew = SlewRateLimiter(c.max_joint_speed)
        self._upper_len: collections.deque = collections.deque(maxlen=150)
        self._fore_len: collections.deque = collections.deque(maxlen=150)
        self._out = JointTargets()
        self._roll_cont = 0.0          # unwrapped wrist roll
        self._last_valid_t: float | None = None

    # -- internal -------------------------------------------------------------

    def _arm_scale(self, upper: float, fore: float) -> float:
        self._upper_len.append(upper)
        self._fore_len.append(fore)
        if len(self._upper_len) >= 10:
            upper = float(np.median(self._upper_len))
            fore = float(np.median(self._fore_len))
        human_reach = upper + fore
        if human_reach < 1e-6:
            return 1.0
        return (self.cfg.l1 + self.cfg.l2) / human_reach

    def _to_robot(self, v_torso: np.ndarray) -> np.ndarray:
        """(right, forward, up) torso components → robot (x fwd, y left, z up)."""
        a, b, c = v_torso
        y = a if self.cfg.mirror else -a
        return np.array([b, self.cfg.base_sign * y, c])

    # -- public ---------------------------------------------------------------

    def update(self, t: float,
               pose_world: Any,
               hand_world: Any = None) -> JointTargets:
        """
        Parameters
        ----------
        t          : monotonic timestamp in seconds.
        pose_world : MediaPipe pose world landmarks (33), or None.
        hand_world : MediaPipe hand world landmarks (21) for the RIGHT hand,
                     or None. (tracking.py guarantees right-hand selection.)
        """
        cfg = self.cfg
        out = self._out

        # Reset filters after a long dropout so stale state doesn't drag.
        if (self._last_valid_t is not None
                and t - self._last_valid_t > cfg.dropout_reset_s):
            self._pose_filter.reset()
            self._hand_filter.reset()
            self._slew.reset()

        # ── Gate ────────────────────────────────────────────────────────────
        if pose_world is None or len(pose_world) < 33:
            out.tracking = False
            out.hand_tracking = False
            return out

        core_ok = all(_vis(pose_world[i]) >= cfg.visibility_min
                      for i in (L_SH, R_SH, R_EL, R_WR))
        if not core_ok:
            out.tracking = False
            out.hand_tracking = False
            return out

        hips_ok = all(_vis(pose_world[i]) >= 0.4 for i in (L_HIP, R_HIP))
        nose_ok = _vis(pose_world[NOSE]) >= 0.4

        # ── Filter landmark positions ──────────────────────────────────────
        raw = np.array([_v(pose_world[i]) for i in _POSE_SUBSET])
        filt = self._pose_filter(raw, t)
        pts = {idx: filt[k] for k, idx in enumerate(_POSE_SUBSET)}

        # ── Torso frame ─────────────────────────────────────────────────────
        M = torso_frame(pts, hips_ok, nose_ok)
        if M is None:
            out.tracking = False
            return out
        up_world = M[2]

        S, E, W = pts[R_SH], pts[R_EL], pts[R_WR]
        upper_t = M @ (E - S)
        wrist_t = M @ (W - S)

        # ── Arm joints ──────────────────────────────────────────────────────
        scale = self._arm_scale(float(np.linalg.norm(E - S)),
                                float(np.linalg.norm(W - E)))
        if cfg.mode == "ik":
            target = self._to_robot(wrist_t) * scale
            q1, q2, q3, target = solve_ik(
                target, cfg.l1, cfg.l2, out.base, cfg.yaw_lock_lo, cfg.yaw_lock_hi)
            out.target_xyz = target
        else:
            q1, q2, q3 = joint_angles_direct(
                self._to_robot(upper_t), S, E, W,
                out.base, cfg.yaw_lock_lo, cfg.yaw_lock_hi)
            out.target_xyz = fk(q1, q2, q3, cfg.l1, cfg.l2)
        out.scale = scale

        # ── Wrist roll + grip (hand landmarks) ─────────────────────────────
        roll = self._roll_cont
        grip = out.grip
        hand_ok = hand_world is not None and len(hand_world) >= 21
        if hand_ok:
            hraw = np.array([_v(hand_world[i]) for i in _HAND_SUBSET])
            hf = self._hand_filter(hraw, t)
            h_wrist, h_thumb, h_imcp, h_itip, h_mmcp, h_pmcp = hf

            r = wrist_roll(W - E, h_imcp - h_pmcp, up_world)
            if r is not None:
                r *= cfg.roll_sign
                # Unwrap against the previous value for continuity at ±180°.
                self._roll_cont += _ang_diff(self._roll_cont, r)
                roll = self._roll_cont

            g = grip_fraction(h_thumb, h_itip, h_wrist, h_mmcp,
                              cfg.pinch_closed, cfg.pinch_open)
            if g is not None:
                grip = g

        # ── Clamp to joint limits, slew-limit, publish ──────────────────────
        q = np.array([
            min(max(q1, cfg.base_range[0]), cfg.base_range[1]),
            min(max(q2, cfg.shoulder_range[0]), cfg.shoulder_range[1]),
            min(max(q3, cfg.elbow_range[0]), cfg.elbow_range[1]),
            min(max(roll, cfg.wrist_range[0]), cfg.wrist_range[1]),
        ])
        q = self._slew(q, t)

        out.base, out.shoulder, out.elbow, out.wrist_roll = (float(v) for v in q)
        out.grip = grip
        out.tracking = True
        out.hand_tracking = hand_ok
        self._last_valid_t = t
        return out
