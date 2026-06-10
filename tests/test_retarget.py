"""
tests/test_retarget.py — Synthetic-pose verification of the retargeting math.

Every sign convention (yaw direction, elevation, elbow flexion, wrist roll,
IK branch) is pinned down here with hand-constructed landmark sets, so a
regression that would scramble the arm in the sim fails loudly on the desk.

Run with either:
    python -m pytest tests/
    python tests/test_retarget.py
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retarget import (ArmRetargeter, RetargetConfig, fk, solve_ik,
                      joint_angles_direct, torso_frame, wrist_roll,
                      NOSE, L_SH, R_SH, R_EL, R_WR, L_HIP, R_HIP,
                      H_WRIST, H_THUMB_TIP, H_INDEX_MCP, H_INDEX_TIP,
                      H_MIDDLE_MCP, H_PINKY_MCP)


# ── Synthetic world ────────────────────────────────────────────────────────────
# Camera-style axes: x = image right, y = image down, z = depth.
# Subject faces the camera, so: subject's right = -x, up = -y, anterior = -z.

def lm(x, y, z, vis=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=vis)


def make_pose(elbow_off, wrist_off):
    """33 pose landmarks for an upright camera-facing subject whose right
    elbow/wrist sit at the given offsets (in world meters) from the right
    shoulder at (-0.18, -0.25, 0)."""
    pts = [lm(0, 0, 0, vis=0.0) for _ in range(33)]
    pts[NOSE] = lm(0.0, -0.55, -0.10)
    pts[L_SH] = lm(+0.18, -0.25, 0.0)
    pts[R_SH] = lm(-0.18, -0.25, 0.0)
    pts[L_HIP] = lm(+0.12, +0.25, 0.0)
    pts[R_HIP] = lm(-0.12, +0.25, 0.0)
    s = np.array([-0.18, -0.25, 0.0])
    e = s + np.asarray(elbow_off, float)
    w = e + np.asarray(wrist_off, float)
    pts[R_EL] = lm(*e)
    pts[R_WR] = lm(*w)
    return pts


def run_frames(pose, hand=None, mode="ik", mirror=True, n=40):
    """Feed the same landmarks repeatedly so filters/slew converge."""
    rt = ArmRetargeter(RetargetConfig(mode=mode, mirror=mirror))
    out = None
    for i in range(n):
        out = rt.update(0.033 * (i + 1), pose, hand)
    return out


def close(a, b, tol_deg=2.0):
    assert abs(math.degrees(a) - math.degrees(b)) < tol_deg, \
        f"expected {math.degrees(b):.1f} deg, got {math.degrees(a):.1f} deg"


# ── Torso frame ────────────────────────────────────────────────────────────────

def test_torso_frame_axes():
    pts_list = make_pose([0, 0.28, 0], [0, 0.27, 0])  # arm hanging down
    pts = {i: np.array([pts_list[i].x, pts_list[i].y, pts_list[i].z])
           for i in (NOSE, L_SH, R_SH, R_EL, R_WR, L_HIP, R_HIP)}
    M = torso_frame(pts, hips_ok=True, nose_ok=True)
    assert M is not None
    assert np.allclose(M[0], [-1, 0, 0], atol=1e-6), f"right axis: {M[0]}"
    assert np.allclose(M[2], [0, -1, 0], atol=1e-6), f"up axis: {M[2]}"
    assert np.allclose(M[1], [0, 0, -1], atol=1e-6), f"anterior axis: {M[1]}"


# ── IK / FK ────────────────────────────────────────────────────────────────────

def test_ik_fk_roundtrip():
    rng = np.random.default_rng(7)
    l1, l2 = 0.30, 0.25
    for _ in range(300):
        q1 = rng.uniform(-2.0, 2.0)
        q2 = rng.uniform(-1.5, 1.5)
        q3 = rng.uniform(0.15, 2.5)
        p = fk(q1, q2, q3, l1, l2)
        if math.hypot(p[0], p[1]) < 0.26 * (l1 + l2):
            continue  # inside the yaw soft-lock band, exactness is traded away
        r1, r2, r3, _ = solve_ik(p, l1, l2, prev_yaw=q1,
                                 lock_lo=0.10, lock_hi=0.25)
        p2 = fk(r1, r2, r3, l1, l2)
        assert np.linalg.norm(p - p2) < 1e-6, (p, p2)
        # Joint-level equality only holds when the sampled config keeps the
        # wrist on the +h side of the base axis (otherwise the equivalent
        # yaw-flipped solution is returned, which is still position-exact).
        h_planar = l1 * math.cos(q2) + l2 * math.cos(q2 + q3)
        if h_planar > 0.05:
            assert abs(r3 - q3) < 1e-6 and abs(r2 - q2) < 1e-6


def test_ik_straight_forward():
    q1, q2, q3, _ = solve_ik(np.array([0.5499, 0, 0]), 0.30, 0.25,
                             prev_yaw=0.0, lock_lo=0.10, lock_hi=0.25)
    close(q1, 0.0)
    close(q2, 0.0, tol_deg=3.0)
    close(q3, 0.0, tol_deg=3.0)


def test_ik_unreachable_target_is_clamped():
    q1, q2, q3, t = solve_ik(np.array([2.0, 0, 0]), 0.30, 0.25,
                             prev_yaw=0.0, lock_lo=0.10, lock_hi=0.25)
    assert np.linalg.norm(t) <= 0.55
    p = fk(q1, q2, q3, 0.30, 0.25)
    assert np.linalg.norm(p - t) < 1e-6


def test_ik_yaw_hold_near_singularity():
    # Target almost straight up: yaw must stay at its previous value.
    q1, *_ = solve_ik(np.array([0.001, 0.001, 0.40]), 0.30, 0.25,
                      prev_yaw=1.0, lock_lo=0.10, lock_hi=0.25)
    close(q1, 1.0)


# ── End-to-end IK mode ─────────────────────────────────────────────────────────

def test_arm_forward_ik():
    # Whole arm pointing straight at the camera (anterior = -z).
    out = run_frames(make_pose([0, 0, -0.28], [0, 0, -0.27]))
    assert out.tracking
    close(out.base, 0.0, tol_deg=3.0)
    close(out.shoulder, 0.0, tol_deg=4.0)
    close(out.elbow, 0.0, tol_deg=4.0)


def test_arm_to_side_mirror_yaw():
    # Arm straight out to the subject's right; mirror mode → robot yaw +90.
    out = run_frames(make_pose([-0.28, 0, 0], [-0.27, 0, 0]), mirror=True)
    close(out.base, math.pi / 2, tol_deg=4.0)
    out = run_frames(make_pose([-0.28, 0, 0], [-0.27, 0, 0]), mirror=False)
    close(out.base, -math.pi / 2, tol_deg=4.0)


def test_arm_raised_ik():
    # Arm straight up: shoulder ≈ +90 (clamped near range edge is fine).
    out = run_frames(make_pose([0, -0.28, 0], [0, -0.27, 0]))
    assert math.degrees(out.shoulder) > 80.0


def test_end_effector_tracks_hand():
    # FK of the commanded joints must land on the IK target (position match).
    out = run_frames(make_pose([0, 0.20, -0.20], [0, -0.05, -0.26]))
    cfg = RetargetConfig()
    p = fk(out.base, out.shoulder, out.elbow, cfg.l1, cfg.l2)
    assert np.linalg.norm(p - out.target_xyz) < 0.01, \
        f"EE {p} vs target {out.target_xyz}"


# ── Joint mode ─────────────────────────────────────────────────────────────────

def test_joint_mode_elbow_bend():
    # Upper arm hanging down, forearm pointing at the camera: 90 deg flexion.
    out = run_frames(make_pose([0, 0.28, 0], [0, 0, -0.27]), mode="joint")
    close(out.elbow, math.pi / 2, tol_deg=4.0)
    close(out.shoulder, -math.pi / 2, tol_deg=4.0)


def test_joint_mode_horizontal_arm():
    out = run_frames(make_pose([0, 0, -0.28], [0, 0, -0.27]), mode="joint")
    close(out.shoulder, 0.0, tol_deg=4.0)
    close(out.elbow, 0.0, tol_deg=4.0)
    close(out.base, 0.0, tol_deg=4.0)


# ── Wrist roll ─────────────────────────────────────────────────────────────────

def test_wrist_roll_signs():
    up = np.array([0.0, -1.0, 0.0])           # torso up in camera axes
    forearm = np.array([0.0, 0.0, -1.0])      # pointing at the camera
    # Neutral handshake: index knuckle on top → pinky→index points up.
    r = wrist_roll(forearm, np.array([0, -1, 0.0]), up)
    close(r, 0.0)
    # Pronated (palm down): right hand, pinky→index points to subject's left (+x).
    r = wrist_roll(forearm, np.array([1, 0, 0.0]), up)
    close(r, -math.pi / 2)
    # Supinated (palm up): pinky→index points to subject's right (-x).
    r = wrist_roll(forearm, np.array([-1, 0, 0.0]), up)
    close(r, math.pi / 2)


# ── Gating ─────────────────────────────────────────────────────────────────────

def test_low_visibility_holds_output():
    pose = make_pose([0, 0, -0.28], [0, 0, -0.27])
    out1 = run_frames(pose)
    rt = ArmRetargeter(RetargetConfig())
    for i in range(10):
        rt.update(0.033 * (i + 1), pose)
    pose[R_EL].visibility = 0.1   # elbow becomes untrusted
    out = rt.update(0.5, pose)
    assert not out.tracking


def test_no_pose_returns_safe_defaults():
    rt = ArmRetargeter(RetargetConfig())
    out = rt.update(0.033, None)
    assert not out.tracking
    assert 0.0 <= out.grip <= 1.0


# ── Plain runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
