from typing import List
import numpy as np


def landmarks_to_pixel_xy(landmarks: List, image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    pixel_points = []
    for lm in landmarks:
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        pixel_points.append((x_px, y_px))
    return np.asarray(pixel_points, dtype=np.int32)
