"""
tracking.py — Camera capture and MediaPipe landmark extraction (Tasks API).

Improvements over the previous version:
  * Requests 1280x720 @ 30fps with MJPG — higher-resolution hand crops give
    measurably better hand landmarks than the 640x480 default.
  * Detects up to TWO hands and selects the subject's RIGHT hand by proximity
    to the pose model's right-wrist landmark (handedness labels alone are
    unreliable because they flip with mirrored capture).
  * Exposes a monotonic timestamp per frame for the downstream filters.
  * Supports the "heavy" pose model for maximum landmark accuracy
    (auto-downloaded on first use).
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision


# ── Model files ────────────────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_POSE_URLS = {
    "full": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    "heavy": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}
_HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def _ensure_model(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[setup] Downloading {os.path.basename(path)} (one-time)...")
    urllib.request.urlretrieve(url, path)
    print(f"[setup] Saved: {path}")


# ── Skeleton connections for the overlay ───────────────────────────────────────

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
_RIGHT_ARM_CONNECTIONS = [(12, 14), (14, 16)]

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class TrackingResult:
    """All tracking data for one frame.

    *_landmarks       : normalized image coordinates (x, y in [0,1])
    *_world_landmarks : metric 3-D coordinates (meters; pose is hip-centered,
                        hand is hand-centered — directions are camera-aligned
                        in both, so vectors are comparable across the two)
    """
    frame: np.ndarray
    timestamp: float
    pose_landmarks: Optional[list] = None
    pose_world_landmarks: Optional[list] = None
    hand_landmarks: Optional[list] = None          # subject's RIGHT hand only
    hand_world_landmarks: Optional[list] = None


# ── Tracker ────────────────────────────────────────────────────────────────────

class ArmTracker:
    """Camera + pose + hand tracking tuned for right-arm teleoperation."""

    def __init__(self,
                 camera_index: int = 0,
                 pose_confidence: float = 0.5,
                 hand_confidence: float = 0.5,
                 pose_model: str = "full",
                 width: int = 1280,
                 height: int = 720):

        pose_path = os.path.join(_MODEL_DIR, f"pose_landmarker_{pose_model}.task")
        hand_path = os.path.join(_MODEL_DIR, "hand_landmarker.task")
        _ensure_model(_POSE_URLS[pose_model], pose_path)
        _ensure_model(_HAND_URL, hand_path)

        self._cap = self._open_camera(camera_index, width, height)
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

        self._pose = _mp_vision.PoseLandmarker.create_from_options(
            _mp_vision.PoseLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=pose_path),
                running_mode=_mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=pose_confidence,
                min_pose_presence_confidence=pose_confidence,
                min_tracking_confidence=pose_confidence,
                output_segmentation_masks=False,
            ))

        self._hands = _mp_vision.HandLandmarker.create_from_options(
            _mp_vision.HandLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=hand_path),
                running_mode=_mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=hand_confidence,
                min_hand_presence_confidence=hand_confidence,
                min_tracking_confidence=hand_confidence,
            ))

    @staticmethod
    def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
        # DirectShow opens faster and more reliably on Windows.
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        cap = None
        for be in backends:
            cap = cv2.VideoCapture(index, be)
            if cap.isOpened():
                break
            cap.release()
        if cap is None or not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}. Make sure it is connected and "
                "not used by another application.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    # ── Right-hand selection ───────────────────────────────────────────────────

    @staticmethod
    def _pick_right_hand(hands_res, pose_landmarks) -> Optional[int]:
        """Index of the detected hand belonging to the subject's right arm.

        Primary cue: image-space distance between each hand's wrist and the
        pose model's right wrist (landmark 16). Falls back to the handedness
        classifier when the pose wrist is unavailable.
        """
        if not hands_res.hand_landmarks:
            return None

        if pose_landmarks is not None and len(pose_landmarks) > 16:
            pw = pose_landmarks[16]
            best, best_d = None, 0.18  # normalized-units acceptance radius
            for i, lms in enumerate(hands_res.hand_landmarks):
                d = ((lms[0].x - pw.x) ** 2 + (lms[0].y - pw.y) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                return best

        for i, hd in enumerate(hands_res.handedness or []):
            if hd and hd[0].category_name == "Right":
                return i
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def read_frame(self) -> TrackingResult:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped delivering frames.")

        t = time.monotonic() - self._t0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing integer milliseconds.
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        pose_res = self._pose.detect_for_video(mp_img, ts_ms)
        hands_res = self._hands.detect_for_video(mp_img, ts_ms)

        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        hand_i = self._pick_right_hand(hands_res, pose_lms)

        return TrackingResult(
            frame=frame,
            timestamp=t,
            pose_landmarks=pose_lms,
            pose_world_landmarks=(pose_res.pose_world_landmarks[0]
                                  if pose_res.pose_world_landmarks else None),
            hand_landmarks=(hands_res.hand_landmarks[hand_i]
                            if hand_i is not None else None),
            hand_world_landmarks=(hands_res.hand_world_landmarks[hand_i]
                                  if hand_i is not None else None),
        )

    def annotate_frame(self, result: TrackingResult) -> np.ndarray:
        """Draw the upper-body skeleton (right arm highlighted) + right hand."""
        frame = result.frame
        h, w = frame.shape[:2]

        def px(lm):
            return int(lm.x * w), int(lm.y * h)

        if result.pose_landmarks:
            lms = result.pose_landmarks
            for a, b in _POSE_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    cv2.line(frame, px(lms[a]), px(lms[b]), (90, 90, 90), 2, cv2.LINE_AA)
            for a, b in _RIGHT_ARM_CONNECTIONS:
                cv2.line(frame, px(lms[a]), px(lms[b]), (0, 220, 80), 3, cv2.LINE_AA)
            for i in (11, 12, 13, 14, 15, 16, 23, 24):
                if i < len(lms):
                    color = (0, 255, 160) if i in (12, 14, 16) else (200, 200, 200)
                    cv2.circle(frame, px(lms[i]), 4, color, -1, cv2.LINE_AA)

        if result.hand_landmarks:
            lms = result.hand_landmarks
            for a, b in _HAND_CONNECTIONS:
                cv2.line(frame, px(lms[a]), px(lms[b]), (255, 180, 30), 2, cv2.LINE_AA)
            for lm in lms:
                cv2.circle(frame, px(lm), 3, (0, 180, 255), -1, cv2.LINE_AA)

        return frame

    @property
    def frame_size(self) -> tuple[int, int]:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self) -> None:
        self._cap.release()
        self._pose.close()
        self._hands.close()

    def __enter__(self) -> "ArmTracker":
        return self

    def __exit__(self, *_) -> None:
        self.release()
