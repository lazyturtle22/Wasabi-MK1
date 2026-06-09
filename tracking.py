"""
tracking.py - Camera capture and MediaPipe skeletal tracking.

Updated for the MediaPipe Tasks API (mediapipe >= 0.10.14).
The mp.solutions namespace was removed in 0.10.14+; this file uses
mediapipe.tasks.python.vision.PoseLandmarker / HandLandmarker instead.

Model .task files are downloaded automatically on first run (~39 MB total)
and cached in the  models/  subdirectory next to this file.

Public interface is identical to the original — TrackingResult and
ArmTracker are fully compatible with kinematics.py and both main*.py files.
"""

from __future__ import annotations

import time
import urllib.request
import pathlib

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision


# ── Model download / cache ────────────────────────────────────────────────────

_MODEL_DIR = pathlib.Path(__file__).parent / "models"

_POSE_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_HANDS_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

_POSE_MODEL  = _MODEL_DIR / "pose_landmarker_lite.task"
_HANDS_MODEL = _MODEL_DIR / "hand_landmarker.task"


def _ensure_models() -> None:
    """Download .task model files if they are not already cached."""
    _MODEL_DIR.mkdir(exist_ok=True)
    if not _POSE_MODEL.exists():
        print("[tracking] Downloading pose model (~30 MB) — one-time setup ...")
        urllib.request.urlretrieve(_POSE_URL, _POSE_MODEL)
        print("[tracking] Pose model ready.")
    if not _HANDS_MODEL.exists():
        print("[tracking] Downloading hand model (~9 MB) — one-time setup ...")
        urllib.request.urlretrieve(_HANDS_URL, _HANDS_MODEL)
        print("[tracking] Hand model ready.")


# ── Landmark list wrapper ─────────────────────────────────────────────────────
# kinematics.py accesses landmarks as  pose_landmarks.landmark[i].x/y/z
# The Tasks API returns a plain list; this wrapper restores .landmark access.

class _LandmarkList:
    """Thin wrapper so Tasks API landmark lists expose a .landmark attribute."""
    __slots__ = ("landmark",)

    def __init__(self, landmarks: list) -> None:
        self.landmark = landmarks


# ── Drawing constants ─────────────────────────────────────────────────────────

_POSE_CONNECTIONS = [
    # Upper body
    (11, 12),
    (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),   # right arm
    (11, 23), (12, 24),   # torso sides
    (23, 24),             # hips
    # Face outline (minimal)
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Legs
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
]

_HAND_CONNECTIONS = [
    (0, 1), (1, 2),  (2, 3),  (3, 4),          # thumb
    (0, 5), (5, 6),  (6, 7),  (7, 8),           # index
    (0, 9), (9, 10), (10, 11),(11, 12),          # middle
    (0, 13),(13, 14),(14, 15),(15, 16),          # ring
    (0, 17),(17, 18),(18, 19),(19, 20),          # pinky
    (5, 9), (9, 13), (13, 17),                   # palm knuckles
]

# Arm landmark indices — drawn in a highlight colour
_ARM_INDICES = {11, 12, 13, 14, 15, 16}


# ── Public data container ─────────────────────────────────────────────────────

@dataclass
class TrackingResult:
    """
    Container returned by ArmTracker.read_frame().

    Attributes
    ----------
    frame          : BGR image ready for annotation / display.
    pose_landmarks : _LandmarkList wrapping full-body pose landmarks, or None.
    hand_landmarks : _LandmarkList wrapping single-hand landmarks, or None.
    """
    frame:          np.ndarray
    pose_landmarks: Optional[Any] = field(default=None)
    hand_landmarks: Optional[Any] = field(default=None)


# ── Main tracker class ────────────────────────────────────────────────────────

class ArmTracker:
    """
    Unified tracker: opens a camera, runs Pose + Hands on every frame,
    and returns a TrackingResult.

    Parameters
    ----------
    camera_index     : OpenCV VideoCapture index (0 = default webcam).
    pose_confidence  : Detection + tracking confidence for Pose.
    hand_confidence  : Detection + tracking confidence for Hands.
    model_complexity : Ignored (kept for API compatibility with original).
    """

    def __init__(self,
                 camera_index:     int   = 0,
                 pose_confidence:  float = 0.6,
                 hand_confidence:  float = 0.6,
                 model_complexity: int   = 1) -> None:

        _ensure_models()

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that it is connected and not in use by another app."
            )

        # Pose landmarker (VIDEO mode = synchronous, frame-by-frame)
        pose_opts = _mp_vision.PoseLandmarkerOptions(
            base_options=_mp_python.BaseOptions(
                model_asset_path=str(_POSE_MODEL)
            ),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=pose_confidence,
            min_pose_presence_confidence=pose_confidence,
            min_tracking_confidence=pose_confidence,
            output_segmentation_masks=False,
        )
        self._pose = _mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # Hand landmarker
        hand_opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(
                model_asset_path=str(_HANDS_MODEL)
            ),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=hand_confidence,
            min_hand_presence_confidence=hand_confidence,
            min_tracking_confidence=hand_confidence,
        )
        self._hands = _mp_vision.HandLandmarker.create_from_options(hand_opts)

        # Monotonic timestamp origin for VIDEO mode (must increase each frame)
        self._t0_ms: int = int(time.monotonic() * 1000)

    # ── Public API ─────────────────────────────────────────────────────────────

    def read_frame(self) -> TrackingResult:
        """
        Grab one frame from the camera, run both detectors, return results.
        Raises RuntimeError if the camera stops delivering frames.
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped delivering frames.")

        timestamp_ms = int(time.monotonic() * 1000) - self._t0_ms

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result  = self._pose.detect_for_video(mp_image, timestamp_ms)
        hands_result = self._hands.detect_for_video(mp_image, timestamp_ms)

        # Wrap landmarks to restore .landmark[i] access expected by kinematics.py
        pose_lm: Optional[_LandmarkList] = None
        if pose_result.pose_landmarks:
            pose_lm = _LandmarkList(pose_result.pose_landmarks[0])

        hand_lm: Optional[_LandmarkList] = None
        if hands_result.hand_landmarks:
            hand_lm = _LandmarkList(hands_result.hand_landmarks[0])

        return TrackingResult(
            frame=frame.copy(),
            pose_landmarks=pose_lm,
            hand_landmarks=hand_lm,
        )

    def annotate_frame(self, result: TrackingResult) -> np.ndarray:
        """
        Draw the skeleton and hand mesh onto result.frame in-place.
        Returns the same frame (mutated) for chaining convenience.
        """
        frame = result.frame
        h, w  = frame.shape[:2]

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            for a, b in _POSE_CONNECTIONS:
                if a < len(lm) and b < len(lm):
                    x1 = int(lm[a].x * w); y1 = int(lm[a].y * h)
                    x2 = int(lm[b].x * w); y2 = int(lm[b].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (60, 180, 60), 2,
                             cv2.LINE_AA)

            for i, l in enumerate(lm):
                cx = int(l.x * w); cy = int(l.y * h)
                color = (0, 255, 140) if i in _ARM_INDICES else (160, 160, 160)
                radius = 5 if i in _ARM_INDICES else 3
                cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)

        if result.hand_landmarks:
            lm = result.hand_landmarks.landmark

            for a, b in _HAND_CONNECTIONS:
                if a < len(lm) and b < len(lm):
                    x1 = int(lm[a].x * w); y1 = int(lm[a].y * h)
                    x2 = int(lm[b].x * w); y2 = int(lm[b].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (200, 80, 255), 2,
                             cv2.LINE_AA)

            for l in lm:
                cx = int(l.x * w); cy = int(l.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 60, 200), -1, cv2.LINE_AA)

        return frame

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the camera stream."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        """Release camera and MediaPipe resources."""
        self._cap.release()
        self._pose.close()
        self._hands.close()

    def __enter__(self) -> "ArmTracker":
        return self

    def __exit__(self, *_) -> None:
        self.release()
