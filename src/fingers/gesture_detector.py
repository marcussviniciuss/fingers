from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

from .hand_types import HandResult
from .finger_counter import count_fingers, _is_finger_up, _is_thumb_up


THUMB_TIP = 4
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18
WRIST = 0


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula o ângulo em graus entre dois vetores"""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def _is_L_gesture(hand: HandResult) -> bool:
    """
    Detecta gesto L: polegar e indicador levantados formando ~90 graus.
    L tem ângulo maior (mais perpendicular) que arminha.
    No L, o polegar está mais horizontal e o indicador mais vertical.
    """
    thumb = _is_thumb_up(hand.pixel_landmarks)
    index = _is_finger_up(hand.pixel_landmarks, INDEX_TIP, INDEX_PIP)
    middle = _is_finger_up(hand.pixel_landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = _is_finger_up(hand.pixel_landmarks, RING_TIP, RING_PIP)
    pinky = _is_finger_up(hand.pixel_landmarks, PINKY_TIP, PINKY_PIP)
    
    if not (thumb and index and not middle and not ring and not pinky):
        return False
    
    thumb_tip = hand.pixel_landmarks[THUMB_TIP]
    thumb_mcp = hand.pixel_landmarks[THUMB_MCP]
    index_tip = hand.pixel_landmarks[INDEX_TIP]
    index_mcp = hand.pixel_landmarks[INDEX_MCP]
    
    thumb_vec = thumb_tip - thumb_mcp
    index_vec = index_tip - index_mcp
    
    angle = _angle_between_vectors(thumb_vec, index_vec)
    
    thumb_horizontal = abs(thumb_vec[0]) > abs(thumb_vec[1]) * 0.8
    index_vertical = abs(index_vec[1]) > abs(index_vec[0]) * 0.8
    
    return (75.0 <= angle <= 130.0) and thumb_horizontal and index_vertical


def _is_gun_gesture(hand: HandResult) -> bool:
    """
    Detecta gesto arminha: polegar e indicador levantados mas mais alinhados.
    Arminha tem ângulo menor que L (mais paralelos).
    Na arminha, ambos apontam mais na mesma direção (indicador para frente).
    """
    thumb = _is_thumb_up(hand.pixel_landmarks)
    index = _is_finger_up(hand.pixel_landmarks, INDEX_TIP, INDEX_PIP)
    middle = _is_finger_up(hand.pixel_landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = _is_finger_up(hand.pixel_landmarks, RING_TIP, RING_PIP)
    pinky = _is_finger_up(hand.pixel_landmarks, PINKY_TIP, PINKY_PIP)
    
    if not (thumb and index and not middle and not ring and not pinky):
        return False
    
    thumb_tip = hand.pixel_landmarks[THUMB_TIP]
    thumb_mcp = hand.pixel_landmarks[THUMB_MCP]
    index_tip = hand.pixel_landmarks[INDEX_TIP]
    index_mcp = hand.pixel_landmarks[INDEX_MCP]
    
    thumb_vec = thumb_tip - thumb_mcp
    index_vec = index_tip - index_mcp
    
    angle = _angle_between_vectors(thumb_vec, index_vec)
    
    return 15.0 <= angle < 75.0


def detect_gestures(hands: list[HandResult]) -> Tuple[Optional[str], Optional[str]]:
    """
    Detecta gestos nas mãos.
    Returns: (gesto_mao_esquerda, gesto_mao_direita)
    - L apenas na mão esquerda
    - Arminha apenas na mão direita
    """
    left_gesture = None
    right_gesture = None
    
    for hand in hands:
        if hand.handedness_label == "Left" and _is_L_gesture(hand):
            left_gesture = "L"
            break
    
    for hand in hands:
        if hand.handedness_label == "Right" and _is_gun_gesture(hand):
            right_gesture = "arminha"
            break
    
    return left_gesture, right_gesture


class GestureImageDisplay:
    def __init__(self, base_path: Path = Path(".")):
        self.base_path = base_path
        self.l_image = None
        self.gun_image = None
        self.current_display = None
        
    def load_images(self):
        """Carrega as imagens dos gestos"""
        l_path = self.base_path / "l.jpg"
        gun_path = self.base_path / "arminha.png"
        
        if l_path.exists():
            self.l_image = cv2.imread(str(l_path))
        if gun_path.exists():
            self.gun_image = cv2.imread(str(gun_path))
    
    def update(self, left_gesture: Optional[str], right_gesture: Optional[str], 
               frame_shape: Tuple[int, int, int]) -> Optional["cv2.Mat"]:
        """
        Atualiza qual imagem exibir baseado nos gestos detectados.
        Retorna a imagem para exibir ou None.
        Prioridade: L na mão esquerda > arminha na mão direita
        """
        if left_gesture == "L" and self.l_image is not None:
            self.current_display = self.l_image
            return self.l_image
        elif right_gesture == "arminha" and self.gun_image is not None:
            self.current_display = self.gun_image
            return self.gun_image
        
        self.current_display = None
        return None
    
    def draw_on_frame(self, frame: "cv2.Mat", overlay_image: Optional["cv2.Mat"]) -> "cv2.Mat":
        """
        Desenha a imagem do gesto sobreposta no frame, centralizada e redimensionada.
        """
        if overlay_image is None:
            return frame
        
        output = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        h_img, w_img = overlay_image.shape[:2]
        
        max_height = int(h_frame * 0.3)
        scale = max_height / h_img
        new_w = int(w_img * scale)
        new_h = int(h_img * scale)
        
        resized = cv2.resize(overlay_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        y_offset = (h_frame - new_h) // 2
        x_offset = (w_frame - new_w) // 2
        
        if y_offset >= 0 and x_offset >= 0:
            output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return output

