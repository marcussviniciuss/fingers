from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class HandResult:
    handedness_label: str
    pixel_landmarks: np.ndarray


@dataclass
class FrameCounts:
    per_hand_counts: List[Tuple[str, int]]
    total_count: int

