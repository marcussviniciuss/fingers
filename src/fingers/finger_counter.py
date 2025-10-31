from __future__ import annotations

from typing import Dict, List, Tuple
from collections import deque
import numpy as np

from .hand_types import HandResult

# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def _is_right_in_image(pixel_landmarks: np.ndarray) -> bool:
    return pixel_landmarks[INDEX_MCP][0] < pixel_landmarks[PINKY_MCP][0]


def _is_thumb_up(pixel_landmarks: np.ndarray) -> bool:
    tip = pixel_landmarks[THUMB_TIP]
    ip = pixel_landmarks[THUMB_IP]
    mcp = pixel_landmarks[THUMB_MCP]
    
    tip_x = float(tip[0])
    ip_x = float(ip[0])
    mcp_x = float(mcp[0])
    
    right_in_image = _is_right_in_image(pixel_landmarks)
    
    if right_in_image:
        return tip_x < ip_x 
    else:
        return tip_x > ip_x 


def _hand_bbox_height(pixel_landmarks: np.ndarray) -> float:
    ys = pixel_landmarks[:, 1]
    return float(ys.max() - ys.min())


def _is_finger_up(pixel_landmarks: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
    tip = pixel_landmarks[tip_idx]
    pip = pixel_landmarks[pip_idx]
    tip_y = float(tip[1])
    pip_y = float(pip[1])
    bbox_h = _hand_bbox_height(pixel_landmarks)
    min_gap = max(4.0, 0.10 * bbox_h)
    min_len = max(6.0, 0.15 * bbox_h)
    seg_len = float(np.linalg.norm(tip - pip))
    return (pip_y - tip_y) > min_gap and seg_len > min_len


def count_fingers(hand: HandResult) -> Tuple[int, Dict[str, bool]]:
    states: Dict[str, bool] = {}

    thumb = _is_thumb_up(hand.pixel_landmarks)
    states["thumb"] = thumb

    names = ["index", "middle", "ring", "pinky"]
    up_count = 1 if thumb else 0
    for name, tip, pip in zip(names, FINGER_TIPS, FINGER_PIPS):
        is_up = _is_finger_up(hand.pixel_landmarks, tip, pip)
        states[name] = is_up
        if is_up:
            up_count += 1

    return up_count, states


class FingerCounter:
    def __init__(self, history_size: int = 5, hysteresis_frames: int = 2) -> None:
        self.history_size = history_size
        self.hysteresis_frames = hysteresis_frames
        self._history: Dict[str, deque[int]] = {
            "Left": deque(maxlen=history_size),
            "Right": deque(maxlen=history_size),
        }
        self._stable_value: Dict[str, int] = {"Left": 0, "Right": 0}
        self._pending_change_count: Dict[str, int] = {"Left": 0, "Right": 0}

        
    def update(self, hands: List[HandResult]) -> Tuple[List[Tuple[str, int]], int]:
        per_hand_counts: List[Tuple[str, int]] = []
        total = 0

        present_labels = [h.handedness_label for h in hands]

        for h in hands:
            count, _ = count_fingers(h)

            dq = self._history.get(h.handedness_label)
            if dq is None:
                dq = deque(maxlen=self.history_size)
                self._history[h.handedness_label] = dq
            dq.append(count)

            stable = self._stable_value.get(h.handedness_label, 0)
            if count != stable:
                self._pending_change_count[h.handedness_label] = self._pending_change_count.get(h.handedness_label, 0) + 1
                if self._pending_change_count[h.handedness_label] >= self.hysteresis_frames:
                    self._stable_value[h.handedness_label] = count
                    self._pending_change_count[h.handedness_label] = 0
            else:
                self._pending_change_count[h.handedness_label] = 0

        for label in present_labels:
            stable = self._stable_value.get(label, 0)
            per_hand_counts.append((label, stable))
            total += stable

        per_hand_counts.sort(key=lambda x: x[0])
        return per_hand_counts, total
