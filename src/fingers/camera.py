import cv2
from typing import Optional

from .config import CAMERA_WIDTH, CAMERA_HEIGHT


class CameraStream:
    def __init__(self, camera_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera de índice {camera_index}")
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def read_frame(self) -> Optional["cv2.Mat"]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
