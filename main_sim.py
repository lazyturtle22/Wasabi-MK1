"""
main_sim.py — Wasabi-MK1 | Phase 1 + Real-Time MuJoCo Simulation

Combines the MediaPipe skeletal tracker with a MuJoCo 3-DoF arm simulator.
Your physical arm drives the simulated robot arm in real time.

Two windows open simultaneously:
  OpenCV window  : live camera feed with skeleton overlay + angle bars
  MuJoCo viewer  : interactive 3-D robot arm (orbit with left-click + drag)

Only the first three degrees of freedom are used to drive the simulation:
  Human Base Pan     →  robot pan   (joint_base)
  Human Shoulder     →  robot lift  (joint_shoulder)
  Human Elbow Tilt   →  robot bend  (joint_elbow)

Wrist pitch, wrist roll, and gripper are computed and displayed in the
sidebar but are not mapped to the simulation (no physical actuator in the
3-DoF model for those joints).

Usage
-----
    pip install mujoco>=3.0.0        # only extra dependency vs main.py
    python main_sim.py

Controls
--------
  Q (OpenCV window)   — quit
  Close MuJoCo window — quit
  Left-drag in MuJoCo — orbit camera
  Scroll in MuJoCo    — zoom
  Right-drag          — pan
"""

from __future__ import annotations

import time

import cv2
import mujoco
import mujoco.viewer

from tracking              import ArmTracker
from kinematics            import joints_to_angles
from filters               import ExponentialSmoothing
from simulation.mujoco_sim   import ArmSimulator
from simulation.angle_mapper import human_to_robot


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX : int   = 0       # 0 = default webcam, 1+ = external
TARGET_FPS   : int   = 30      # target loop rate for tracking + display
USE_LEFT_ARM : bool  = False   # False = subject's RIGHT arm drives the robot
SMOOTH_ALPHA : float = 0.25    # EMA smoothing — lower = smoother but laggier
POSE_CONF    : float = 0.60    # MediaPipe Pose detection confidence
HAND_CONF    : float = 0.60    # MediaPipe Hands detection confidence

# Physics substeps per control call.  20 × 0.002 s = 0.04 s of sim per frame.
SIM_SUBSTEPS : int = 20


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS  (mirrors main.py sidebar / HUD)
# ══════════════════════════════════════════════════════════════════════════════

_SERVO_LABELS: list[str] = [
    "Forearm Yaw",   # DoF 1 → robot base pan
    "Wrist Tilt",    # DoF 2 → robot shoulder
    "Wrist Roll",    # DoF 3 → robot elbow
    "—", "—", "—",  # unused
]

_BAR_COLORS: list[tuple[int, int, int]] = [
    (0,   200, 255),   # Forearm Yaw — amber-yellow
    (0,   255, 160),   # Wrist Tilt  — mint green
    (200,  80, 255),   # Wrist Roll  — violet
    (60,   60,  60),   # unused
    (60,   60,  60),   # unused
    (60,   60,  60),   # unused
]

# Mark the 3 simulated joints with a highlighted border in the sidebar
_SIM_JOINTS: set[int] = {0, 1, 2}   # indices into _SERVO_LABELS

SIDEBAR_W: int = 240


def _draw_sidebar(canvas: "np.ndarray", angles: list[int]) -> None:
    import numpy as np
    h, w = canvas.shape[:2]
    x0   = w - SIDEBAR_W

    overlay = canvas[:, x0:].copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_W, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, canvas[:, x0:], 0.28, 0, canvas[:, x0:])

    cv2.putText(canvas, "WASABI-MK1  [ SIM ]", (x0 + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Live Servo Angles  (*=sim)", (x0 + 8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

    usable_h = h - 58
    slot_h   = usable_h // len(angles)
    bar_max  = SIDEBAR_W - 22

    for i, (label, angle, color) in enumerate(zip(_SERVO_LABELS, angles, _BAR_COLORS)):
        y0      = 50 + i * slot_h
        max_ang = 180
        bar_w   = int(bar_max * angle / max_ang)
        star    = "*" if i in _SIM_JOINTS else " "

        cv2.putText(canvas, f"{star}{label}: {angle}°",
                    (x0 + 8, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # Track background
        cv2.rectangle(canvas,
                      (x0 + 8,           y0 + 18),
                      (x0 + 8 + bar_max, y0 + 30),
                      (55, 55, 55), -1)

        # Value bar
        if bar_w > 0:
            cv2.rectangle(canvas,
                          (x0 + 8,          y0 + 18),
                          (x0 + 8 + bar_w,  y0 + 30),
                          color, -1)

        # Highlight border for simulated joints
        if i in _SIM_JOINTS:
            cv2.rectangle(canvas,
                          (x0 + 7,           y0 + 17),
                          (x0 + 9 + bar_max, y0 + 31),
                          color, 1)


def _draw_hud(canvas: "np.ndarray",
              fps:      float,
              tracking: bool,
              ee_pos:   "np.ndarray | None" = None) -> None:
    status_color = (0, 220, 80) if tracking else (30, 100, 220)
    status_label = "TRACKING" if tracking else "SEARCHING"

    cv2.putText(canvas, f"FPS {fps:5.1f}   [{status_label}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                status_color, 2, cv2.LINE_AA)

    if ee_pos is not None:
        ee_txt = f"EE  x={ee_pos[0]:+.3f}  y={ee_pos[1]:+.3f}  z={ee_pos[2]:+.3f} m"
        cv2.putText(canvas, ee_txt, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    tracker  = ArmTracker(
        camera_index=CAMERA_INDEX,
        pose_confidence=POSE_CONF,
        hand_confidence=HAND_CONF,
    )
    smoother = ExponentialSmoothing(alpha=SMOOTH_ALPHA, num_channels=6)
    sim      = ArmSimulator(substeps=SIM_SUBSTEPS)

    last_angles: list[int] = [90, 90, 90, 90, 90, 45]   # neutral fallback
    frame_period = 1.0 / TARGET_FPS
    prev_t       = time.perf_counter()
    fps_display  = 0.0

    print("=" * 58)
    print("  Wasabi-MK1 | Phase 1 + MuJoCo 3-DoF Simulation")
    print(f"  Camera: {CAMERA_INDEX}  |  Target: {TARGET_FPS} FPS  "
          f"|  EMA α: {SMOOTH_ALPHA}")
    print("  Control: hold RIGHT forearm vertical (elbow down, wrist up)")
    print("  DoF1=Forearm Yaw  DoF2=Wrist Tilt  DoF3=Wrist Roll  (* in sidebar)")
    print("  Press  Q  in the OpenCV window or close MuJoCo to quit.")
    print("=" * 58)

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            # Set a comfortable default camera pose
            viewer.cam.azimuth   = 140.0
            viewer.cam.elevation = -22.0
            viewer.cam.distance  =  1.40
            viewer.cam.lookat[:] = [0.0, 0.0, 0.32]

            while viewer.is_running():
                t_start = time.perf_counter()

                # ── 1. Capture frame & run detectors ──────────────────────
                result = tracker.read_frame()

                # ── 2. Kinematic mapping ───────────────────────────────────
                raw = joints_to_angles(
                    result.pose_landmarks,
                    result.hand_landmarks,
                    use_left_arm=USE_LEFT_ARM,
                )
                tracking = raw is not None
                if tracking:
                    smoothed    = smoother.update(raw)
                    last_angles = [int(round(v)) for v in smoothed]
                # No detection → hold last_angles so the robot stays put.

                # ── 3. Drive MuJoCo simulation (3 DoF) ───────────────────
                base_r, shoulder_r, elbow_r = human_to_robot(
                    last_angles[0],   # Base Pan
                    last_angles[1],   # Shoulder Tilt
                    last_angles[2],   # Elbow Tilt
                )
                sim.set_target(base_r, shoulder_r, elbow_r)
                sim.step()
                viewer.sync()        # push new state to the 3-D window

                # ── 4. Annotate OpenCV frame ───────────────────────────────
                tracker.annotate_frame(result)
                _draw_sidebar(result.frame, last_angles)

                now         = time.perf_counter()
                fps_display = 1.0 / max(now - prev_t, 1e-6)
                prev_t      = now
                _draw_hud(result.frame, fps_display, tracking, sim.ee_position)

                cv2.imshow("Wasabi-MK1 | Tracking → Sim", result.frame)

                # ── 5. Rate-limit to TARGET_FPS ────────────────────────────
                elapsed = time.perf_counter() - t_start
                wait_ms = max(1, int((frame_period - elapsed) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")

    finally:
        tracker.release()
        cv2.destroyAllWindows()
        print("\nWasabi-MK1 simulation shutdown complete.")


if __name__ == "__main__":
    main()
