from __future__ import annotations

from typing import List, Tuple
import cv2
import mediapipe as mp

from .config import (
    DRAW_CONNECTIONS,
    DRAW_LANDMARKS,
    TEXT_BG_COLOR_BGR,
    TEXT_COLOR_BGR,
    TEXT_SCALE,
    TEXT_THICKNESS,
    MARGIN_PX,
)
from .hand_types import HandResult


_mp_draw = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


def _draw_label(frame, text: str, org: Tuple[int, int]) -> None:
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - h - 6), (x + w + 4, y + baseline + 4), TEXT_BG_COLOR_BGR, -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR_BGR, TEXT_THICKNESS, cv2.LINE_AA)


def draw_hands_and_overlays(
    frame,
    hand_results: List[HandResult],
    per_hand_counts: List[Tuple[str, int]],
    total_count: int,
):
    output = frame.copy()

    if DRAW_LANDMARKS or DRAW_CONNECTIONS:
        for hand in hand_results:
            connections = mp.solutions.hands.HAND_CONNECTIONS

            if DRAW_LANDMARKS:
                for (x, y) in hand.pixel_landmarks:
                    cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            if DRAW_CONNECTIONS:
                for a_idx, b_idx in connections:
                    ax, ay = hand.pixel_landmarks[a_idx]
                    bx, by = hand.pixel_landmarks[b_idx]
                    cv2.line(output, (int(ax), int(ay)), (int(bx), int(by)), (0, 200, 255), 1, lineType=cv2.LINE_AA)

    _draw_label(output, f"Total: {total_count}", (MARGIN_PX, 30))

    line_y = 60
    for label, count in per_hand_counts:
        _draw_label(output, f"{label}: {count}", (MARGIN_PX, line_y))
        line_y += 30

    return output
