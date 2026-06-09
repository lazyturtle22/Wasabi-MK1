"""
main.py — Wasabi-MK1 | Phase 1: PC Vision & Telemetry Engine

Orchestrates the full pipeline:
  Camera → MediaPipe tracking → kinematic angle mapping
  → EMA smoothing → telemetry packet → console (+ optional UDP/TCP stub)
  → OpenCV overlay with live angle sidebar

Usage:
    python main.py

Press  Q  in the OpenCV window to quit.
"""

from __future__ import annotations

import socket
import time

import cv2
import numpy as np

from tracking   import ArmTracker
from kinematics import joints_to_angles
from filters    import ExponentialSmoothing


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these constants to tune behaviour
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX   : int   = 0      # 0 = built-in webcam, 1+ = external camera
TARGET_FPS     : int   = 30     # target telemetry + display loop rate
USE_LEFT_ARM   : bool  = True   # True  → track subject's LEFT arm (mirror mode)
                                 # False → track subject's RIGHT arm
SMOOTH_ALPHA   : float = 0.25   # EMA smoothing factor: 0 (frozen) → 1 (raw)
POSE_CONF      : float = 0.60   # MediaPipe Pose confidence threshold
HAND_CONF      : float = 0.60   # MediaPipe Hands confidence threshold

# ── Network target (fill in when Wi-Fi hardware arrives) ──────────────────────
ROBOT_IP   : str = "192.168.1.100"
ROBOT_PORT : int = 5005


# ══════════════════════════════════════════════════════════════════════════════
#  TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════

def format_packet(angles: list[int]) -> str:
    """
    Serialise the six servo angles into a compact string packet.

    Format:  <BASE,SHOULDER,ELBOW,WRIST_P,WRIST_R,GRIPPER>\n

    This exact format is what the Arduino sketch will parse with sscanf or
    strtok once the Wi-Fi link is active.
    """
    return "<{},{},{},{},{},{}>\n".format(*angles)


def send_packet(packet: str) -> None:
    """
    Transmit (or simulate) one telemetry packet.

    Phase 1: prints to the console.
    Phase 2: uncomment the UDP block below and comment out the print().
    """
    # ── Console output (Phase 1) ───────────────────────────────────────────
    print(packet, end="", flush=True)

    # ── UDP transmission (uncomment for Phase 2) ───────────────────────────
    # try:
    #     with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    #         sock.sendto(packet.encode("ascii"), (ROBOT_IP, ROBOT_PORT))
    # except OSError as e:
    #     print(f"[UDP] send error: {e}")

    # ── TCP transmission alternative (uncomment for Phase 2) ──────────────
    # try:
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    #         sock.settimeout(0.05)                       # 50 ms timeout
    #         sock.connect((ROBOT_IP, ROBOT_PORT))
    #         sock.sendall(packet.encode("ascii"))
    # except OSError as e:
    #     print(f"[TCP] send error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

_SERVO_LABELS: list[str] = [
    "Base Pan",
    "Shoulder",
    "Elbow",
    "Wrist Pitch",
    "Wrist Roll",
    "Gripper",
]

_BAR_COLORS: list[tuple[int, int, int]] = [
    (0,   200, 255),   # Base Pan    — amber-yellow
    (0,   255, 160),   # Shoulder    — mint green
    (0,   255,  60),   # Elbow       — bright green
    (80,  200, 255),   # Wrist Pitch — sky blue
    (200,  80, 255),   # Wrist Roll  — violet
    (255, 160,  60),   # Gripper     — orange
]

SIDEBAR_W: int = 230   # pixel width of the right-hand info panel


def _draw_sidebar(canvas: np.ndarray, angles: list[int]) -> None:
    """
    Render a semi-transparent dark sidebar on the right of *canvas* showing
    a labelled progress bar for each servo angle.
    """
    h, w = canvas.shape[:2]
    x0 = w - SIDEBAR_W

    # Dark overlay
    overlay = canvas[:, x0:].copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_W, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, canvas[:, x0:], 0.30, 0, canvas[:, x0:])

    # Header
    cv2.putText(canvas, "WASABI-MK1", (x0 + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Live Servo Angles", (x0 + 8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

    usable_h = h - 60
    slot_h   = usable_h // len(angles)
    bar_max  = SIDEBAR_W - 20

    for i, (label, angle, color) in enumerate(zip(_SERVO_LABELS, angles, _BAR_COLORS)):
        y0      = 55 + i * slot_h
        max_ang = 90 if label == "Gripper" else 180
        bar_w   = int(bar_max * angle / max_ang)

        # Label + numeric value
        cv2.putText(canvas, f"{label}: {angle}°",
                    (x0 + 8, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # Track (background)
        cv2.rectangle(canvas,
                      (x0 + 8,          y0 + 18),
                      (x0 + 8 + bar_max, y0 + 30),
                      (55, 55, 55), -1)

        # Value bar
        if bar_w > 0:
            cv2.rectangle(canvas,
                          (x0 + 8,          y0 + 18),
                          (x0 + 8 + bar_w,  y0 + 30),
                          color, -1)


def _draw_hud(canvas: np.ndarray, fps: float, tracking: bool) -> None:
    """Render top-left HUD showing FPS and tracking status."""
    color = (0, 220, 80) if tracking else (30, 100, 220)
    label = "TRACKING" if tracking else "SEARCHING"
    cv2.putText(canvas, f"FPS {fps:5.1f}   [{label}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


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

    # Safe fallback angles (neutral pose) used during tracking dropout
    last_angles: list[int] = [90, 90, 90, 90, 90, 45]

    frame_period = 1.0 / TARGET_FPS
    prev_t       = time.perf_counter()
    fps_display  = 0.0

    print("=" * 52)
    print("  Wasabi-MK1 | Phase 1 — PC Vision Engine")
    print(f"  Camera: {CAMERA_INDEX}  |  Target: {TARGET_FPS} FPS  "
          f"|  EMA α: {SMOOTH_ALPHA}")
    print("  Press  Q  in the OpenCV window to quit.")
    print("=" * 52)

    try:
        while True:
            t_start = time.perf_counter()

            # 1. Capture + detect
            result = tracker.read_frame()

            # 2. Kinematic mapping
            raw = joints_to_angles(
                result.pose_landmarks,
                result.hand_landmarks,
                use_left_arm=USE_LEFT_ARM,
            )

            tracking = raw is not None
            if tracking:
                smoothed    = smoother.update(raw)
                last_angles = [int(round(v)) for v in smoothed]
            # If no detection, hold last_angles and let servos stay put.

            # 3. Transmit telemetry packet
            packet = format_packet(last_angles)
            send_packet(packet)

            # 4. Annotate frame + render sidebar
            tracker.annotate_frame(result)
            _draw_sidebar(result.frame, last_angles)

            now         = time.perf_counter()
            fps_display = 1.0 / max(now - prev_t, 1e-6)
            prev_t      = now
            _draw_hud(result.frame, fps_display, tracking)

            cv2.imshow("Wasabi-MK1 | Phase 1", result.frame)

            # 5. Rate-limit to TARGET_FPS
            elapsed  = time.perf_counter() - t_start
            wait_ms  = max(1, int((frame_period - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")

    finally:
        tracker.release()
        cv2.destroyAllWindows()
        print("\nWasabi-MK1 shutdown complete.")


if __name__ == "__main__":
    main()
