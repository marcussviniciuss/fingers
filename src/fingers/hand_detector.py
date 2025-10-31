from __future__ import annotations

from typing import List
import cv2
import mediapipe as mp

from .config import (
    MAX_NUM_HANDS,
    MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from .hand_types import HandResult
from .utils import landmarks_to_pixel_xy


class HandDetector:
    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def detect_hands(self, bgr_frame: "cv2.Mat") -> List[HandResult]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        hands: List[HandResult] = []

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" ou "Right"
                pixel_lms = landmarks_to_pixel_xy(hand_landmarks.landmark, bgr_frame.shape)
                hands.append(HandResult(handedness_label=label, pixel_landmarks=pixel_lms))

        return hands

    def close(self) -> None:
        try:
            if self._hands is not None:
                self._hands.close()
        except Exception:
            pass
