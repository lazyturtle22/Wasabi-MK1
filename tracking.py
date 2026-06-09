"""
tracking.py - Camera capture and MediaPipe skeletal tracking.

Wraps OpenCV VideoCapture and two MediaPipe pipelines (Pose + Hands) into a
single class so that main.py stays clean.  Only one hand is tracked (the
operator hand used for gripper / wrist control).
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


# MediaPipe solution handles
_mp_pose        = mp.solutions.pose
_mp_hands       = mp.solutions.hands
_mp_draw        = mp.solutions.drawing_utils
_mp_draw_styles = mp.solutions.drawing_styles


@dataclass
class TrackingResult:
    """
    Container returned by ArmTracker.read_frame().

    Attributes
    ----------
    frame          : BGR image ready for annotation / display.
    pose_landmarks : Full-body pose landmarks, or None.
    hand_landmarks : First detected hand landmarks, or None.
    """
    frame:          np.ndarray
    pose_landmarks: Optional[Any] = field(default=None)
    hand_landmarks: Optional[Any] = field(default=None)


class ArmTracker:
    """
    Unified tracker: opens a camera, runs Pose + Hands on every frame,
    and returns a TrackingResult.

    Parameters
    ----------
    camera_index     : OpenCV VideoCapture index (0 = default webcam).
    pose_confidence  : Detection + tracking confidence threshold for Pose.
    hand_confidence  : Detection + tracking confidence threshold for Hands.
    model_complexity : MediaPipe Pose model complexity (0 = fast, 2 = accurate).
    """

    def __init__(self,
                 camera_index:     int   = 0,
                 pose_confidence:  float = 0.6,
                 hand_confidence:  float = 0.6,
                 model_complexity: int   = 1):

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that the camera is connected and not used by another app."
            )

        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=pose_confidence,
            min_tracking_confidence=pose_confidence,
        )

        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=hand_confidence,
            min_tracking_confidence=hand_confidence,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def read_frame(self) -> TrackingResult:
        """
        Grab one frame from the camera, run both detectors, return results.
        Raises RuntimeError if the camera stops delivering frames.
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped delivering frames.")

        # Convert once to RGB; both MediaPipe models expect RGB input.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # avoids an internal copy in MediaPipe

        pose_result  = self._pose.process(rgb)
        hands_result = self._hands.process(rgb)

        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Pull the first detected hand (we only care about one)
        hand_lm: Optional[Any] = None
        if hands_result.multi_hand_landmarks:
            hand_lm = hands_result.multi_hand_landmarks[0]

        return TrackingResult(
            frame=annotated,
            pose_landmarks=pose_result.pose_landmarks,
            hand_landmarks=hand_lm,
        )

    def annotate_frame(self, result: TrackingResult) -> np.ndarray:
        """
        Draw the skeleton and hand mesh onto result.frame in-place.
        Returns the same frame (mutated) for chaining convenience.
        """
        if result.pose_landmarks:
            _mp_draw.draw_landmarks(
                result.frame,
                result.pose_landmarks,
                _mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_draw_styles.get_default_pose_landmarks_style(),
            )

        if result.hand_landmarks:
            _mp_draw.draw_landmarks(
                result.frame,
                result.hand_landmarks,
                _mp_hands.HAND_CONNECTIONS,
                _mp_draw_styles.get_default_hand_landmarks_style(),
                _mp_draw_styles.get_default_hand_connections_style(),
            )

        return result.frame

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

    # Allow use as a context manager: `with ArmTracker() as t:`
    def __enter__(self) -> "ArmTracker":
        return self

    def __exit__(self, *_) -> None:
        self.release()
