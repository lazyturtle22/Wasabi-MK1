"""
main.py — Wasabi-MK1: camera-to-simulation right-arm teleoperation.

Pipeline per frame:
  Camera → MediaPipe (pose + right hand) → One-Euro filtering
  → torso-frame retargeting (analytic IK by default) → MuJoCo sim
  → annotated camera preview (+ optional UDP telemetry for the real arm)

Usage:
    python main.py                     # default: IK mode, mirrored, dynamics sim
    python main.py --mode joint        # pose-mimicry instead of hand tracking
    python main.py --kinematic         # zero-lag exact sim (precision check)
    python main.py --pose-model heavy  # most accurate pose model (slower)
    python main.py --no-mirror         # non-mirrored mapping
    python main.py --udp 192.168.1.100:5005   # also stream servo packets
    python main.py --no-sim            # camera/overlay only, no MuJoCo window

Press Q or ESC in the camera window to quit.
"""

from __future__ import annotations

import argparse
import math
import socket
import time

import cv2
import numpy as np

from tracking import ArmTracker
from retarget import ArmRetargeter, RetargetConfig, JointTargets


# ── Telemetry for the future real arm ─────────────────────────────────────────

def servo_packet(t: JointTargets) -> str:
    """
    Map joint targets to the 6-servo packet the Arduino sketch will parse:
        <BASE,SHOULDER,ELBOW,WRIST_P,WRIST_R,GRIPPER>\n
    All values are integer servo degrees. Wrist pitch has no DoF on this arm
    and is held at 90.
    """
    def clamp(v):
        return int(np.clip(round(v), 0, 180))
    base     = clamp(90 + math.degrees(t.base))
    shoulder = clamp(90 + math.degrees(t.shoulder))
    elbow    = clamp(180 - math.degrees(t.elbow))      # 180 = straight
    wrist_r  = clamp(90 + math.degrees(t.wrist_roll) / 2.0)
    gripper  = int(np.clip(round(t.grip * 90), 0, 90))
    return f"<{base},{shoulder},{elbow},90,{wrist_r},{gripper}>\n"


class UdpSender:
    def __init__(self, addr: str):
        host, port = addr.rsplit(":", 1)
        self._target = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, packet: str) -> None:
        try:
            self._sock.sendto(packet.encode("ascii"), self._target)
        except OSError:
            pass


# ── Overlay HUD ────────────────────────────────────────────────────────────────

_CHANNELS = [
    # label, value-fn (degrees / percent), display range
    ("Base Yaw",   lambda t: math.degrees(t.base),       (-120, 120)),
    ("Shoulder",   lambda t: math.degrees(t.shoulder),   (-95, 95)),
    ("Elbow",      lambda t: math.degrees(t.elbow),      (0, 150)),
    ("Wrist Roll", lambda t: math.degrees(t.wrist_roll), (-180, 180)),
    ("Grip",       lambda t: t.grip * 100,               (0, 100)),
]

_BAR_COLORS = [(0, 200, 255), (0, 255, 160), (0, 255, 60),
               (200, 80, 255), (255, 160, 60)]

SIDEBAR_W = 240


def draw_sidebar(canvas: np.ndarray, t: JointTargets, fps: float,
                 mode: str) -> None:
    h, w = canvas.shape[:2]
    x0 = w - SIDEBAR_W
    overlay = canvas[:, x0:].copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_W, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, canvas[:, x0:], 0.28, 0, canvas[:, x0:])

    cv2.putText(canvas, "WASABI-MK1", (x0 + 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)

    pose_c = (0, 220, 80) if t.tracking else (40, 90, 230)
    hand_c = (0, 220, 80) if t.hand_tracking else (40, 90, 230)
    cv2.putText(canvas, f"POSE {'LOCK' if t.tracking else '----'}",
                (x0 + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pose_c, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"HAND {'LOCK' if t.hand_tracking else '----'}",
                (x0 + 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hand_c, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{fps:4.1f} fps   mode: {mode}",
                (x0 + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

    bar_max = SIDEBAR_W - 24
    slot_h = (h - 90) // len(_CHANNELS)
    for i, ((label, fn, (lo, hi)), color) in enumerate(zip(_CHANNELS, _BAR_COLORS)):
        val = fn(t)
        y0 = 88 + i * slot_h
        unit = "%" if label == "Grip" else "deg"
        cv2.putText(canvas, f"{label}: {val:+6.1f} {unit}",
                    (x0 + 12, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + 12, y0 + 20), (x0 + 12 + bar_max, y0 + 31),
                      (55, 55, 55), -1)
        frac = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
        # Signed channels fill from center; unsigned from the left edge.
        if lo < 0:
            cx = x0 + 12 + bar_max // 2
            px = x0 + 12 + int(bar_max * frac)
            cv2.rectangle(canvas, (min(cx, px), y0 + 20), (max(cx, px), y0 + 31),
                          color, -1)
            cv2.line(canvas, (cx, y0 + 18), (cx, y0 + 33), (120, 120, 120), 1)
        else:
            cv2.rectangle(canvas, (x0 + 12, y0 + 20),
                          (x0 + 12 + int(bar_max * frac), y0 + 31), color, -1)


# ── Main loop ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wasabi-MK1 right-arm teleoperation")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--mode", choices=("ik", "joint"), default="ik",
                   help="ik = track hand position (precise), joint = mimic pose")
    p.add_argument("--no-mirror", action="store_true",
                   help="disable mirror-image mapping")
    p.add_argument("--kinematic", action="store_true",
                   help="drive sim joints directly (zero lag, no servo dynamics)")
    p.add_argument("--pose-model", choices=("full", "heavy"), default="full",
                   help="'heavy' is the most accurate pose model (slower)")
    p.add_argument("--no-sim", action="store_true", help="skip the MuJoCo window")
    p.add_argument("--udp", metavar="HOST:PORT", default=None,
                   help="stream servo packets over UDP (for the real arm)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tracker = ArmTracker(camera_index=args.camera, pose_model=args.pose_model)
    retargeter = ArmRetargeter(RetargetConfig(mode=args.mode,
                                              mirror=not args.no_mirror))
    sim = None
    if not args.no_sim:
        from simulation import ArmSim
        sim = ArmSim(kinematic=args.kinematic)

    udp = UdpSender(args.udp) if args.udp else None

    print("=" * 56)
    print("  Wasabi-MK1 — right-arm teleoperation")
    print(f"  mode: {args.mode}   mirror: {not args.no_mirror}   "
          f"sim: {'kinematic' if args.kinematic else 'dynamics' if sim else 'off'}")
    print("  Face the camera, whole right arm + hips in frame.")
    print("  Press Q or ESC in the camera window to quit.")
    print("=" * 56)

    fps, prev_t = 0.0, time.perf_counter()
    try:
        while True:
            result = tracker.read_frame()
            targets = retargeter.update(result.timestamp,
                                        result.pose_world_landmarks,
                                        result.hand_world_landmarks)

            if sim is not None:
                if not sim.alive:
                    break
                sim.update(targets)

            if udp is not None and targets.tracking:
                udp.send(servo_packet(targets))

            tracker.annotate_frame(result)
            display = cv2.flip(result.frame, 1)   # mirror preview only
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            draw_sidebar(display, targets, fps, args.mode)
            cv2.imshow("Wasabi-MK1 | camera", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
    finally:
        tracker.release()
        if sim is not None:
            sim.close()
        cv2.destroyAllWindows()
        print("\nWasabi-MK1 shutdown complete.")


if __name__ == "__main__":
    main()
