from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter


class EmotionDetector:
    def __init__(self, history_size: int = 7):
        self._history = deque(maxlen=history_size)
        self._last_emotion = "normal"
        self._last_bbox = None
        self._face_mesh = None
        
        try:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"AVISO: Não foi possível inicializar Face Mesh: {e}")
            print("Detecção de emoções desabilitada.")
            self._face_mesh = None
        
    def _landmarks_to_pixel(self, landmarks, image_shape):
        """Converte landmarks normalizados para pixels"""
        height, width = image_shape[:2]
        pixel_points = []
        for lm in landmarks:
            x_px = int(lm.x * width)
            y_px = int(lm.y * height)
            pixel_points.append((x_px, y_px))
        return np.array(pixel_points)
    
    def detect_emotion(self, bgr_frame) -> tuple[Optional[str], Optional[tuple[int, int, int, int]]]:
        """
        Detecta a emoção no frame usando análise avançada de landmarks.
        Returns: (emotion, (x, y, width, height)) ou (None, None) se não detectar rosto
        """
        if self._face_mesh is None:
            return self._last_emotion, self._last_bbox
        
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb)
            
            if not result.multi_face_landmarks:
                return self._last_emotion, self._last_bbox
        except Exception as e:
            return self._last_emotion, self._last_bbox
        
        face_landmarks = result.multi_face_landmarks[0]
        pixel_landmarks = self._landmarks_to_pixel(face_landmarks.landmark, bgr_frame.shape)
        
        x_coords = pixel_landmarks[:, 0]
        y_coords = pixel_landmarks[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(bgr_frame.shape[1], x_max + padding)
        y_max = min(bgr_frame.shape[0], y_max + padding)
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        emotion = self._analyze_emotion_advanced(pixel_landmarks)
        
        self._history.append(emotion)
        stable_emotion = self._get_stable_emotion()
        
        if stable_emotion:
            self._last_emotion = stable_emotion
            self._last_bbox = bbox
        
        return self._last_emotion, self._last_bbox
    
    def _analyze_emotion_advanced(self, landmarks: np.ndarray) -> str:
        """Análise avançada usando múltiplos pontos e relações geométricas"""
        try:
            MOUTH_LEFT = 61
            MOUTH_RIGHT = 291
            MOUTH_TOP = 13
            MOUTH_BOTTOM = 14
            MOUTH_CENTER = 13 
            
            LEFT_EYEBROW_INNER = 107
            RIGHT_EYEBROW_INNER = 336
            LEFT_EYE_TOP = 159
            RIGHT_EYE_TOP = 386
            LEFT_EYE_BOTTOM = 145
            RIGHT_EYE_BOTTOM = 374
            
            NOSE_TIP = 1
            
            mouth_left = landmarks[MOUTH_LEFT]
            mouth_right = landmarks[MOUTH_RIGHT]
            mouth_top = landmarks[MOUTH_TOP]
            mouth_bottom = landmarks[MOUTH_BOTTOM]
            
            left_eyebrow = landmarks[LEFT_EYEBROW_INNER]
            right_eyebrow = landmarks[RIGHT_EYEBROW_INNER]
            left_eye_top = landmarks[LEFT_EYE_TOP]
            right_eye_top = landmarks[RIGHT_EYE_TOP]
            left_eye_bottom = landmarks[LEFT_EYE_BOTTOM]
            right_eye_bottom = landmarks[RIGHT_EYE_BOTTOM]
            nose = landmarks[NOSE_TIP]
            
            face_width = abs(landmarks[:, 0].max() - landmarks[:, 0].min())
            face_height = abs(landmarks[:, 1].max() - landmarks[:, 1].min())
            face_size = max(face_width, face_height)
            
            if face_size < 10:
                return "normal"
            
            mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2.0
            mouth_corners_y = (mouth_left[1] + mouth_right[1]) / 2.0
            mouth_curve = (mouth_center_y - mouth_corners_y) / face_size * 100
            
            mouth_width = abs(mouth_right[0] - mouth_left[0]) / face_size * 100
            mouth_height = abs(mouth_bottom[1] - mouth_top[1]) / face_size * 100
            mouth_aspect = mouth_height / (mouth_width + 1e-6)
            
            left_eyebrow_eye_dist = abs(left_eyebrow[1] - left_eye_top[1]) / face_size * 100
            right_eyebrow_eye_dist = abs(right_eyebrow[1] - right_eye_top[1]) / face_size * 100
            avg_eyebrow_dist = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2.0
            
            eyebrow_center_y = (left_eyebrow[1] + right_eyebrow[1]) / 2.0
            eye_center_y = (left_eye_top[1] + right_eye_top[1]) / 2.0
            eyebrow_drop = (eyebrow_center_y - eye_center_y) / face_size * 100
            
            left_eye_open = abs(left_eye_top[1] - left_eye_bottom[1]) / face_size * 100
            right_eye_open = abs(right_eye_top[1] - right_eye_bottom[1]) / face_size * 100
            avg_eye_open = (left_eye_open + right_eye_open) / 2.0
            
            scores = {
                "feliz": 0.0,
                "triste": 0.0,
                "brava": 0.0,
                "normal": 0.0
            }
            
            if mouth_curve > 2.5:
                scores["feliz"] += 3.0
            if mouth_curve > 4.0:
                scores["feliz"] += 2.0
            if avg_eye_open < 4.5:
                scores["feliz"] += 2.5
            if mouth_aspect > 0.18:
                scores["feliz"] += 1.5
            if mouth_curve > 3.0 and avg_eye_open < 5.0:
                scores["feliz"] += 2.0
            
            if mouth_curve < -3.5:
                scores["triste"] += 5.0
            if mouth_curve < -2.5:
                scores["triste"] += 3.0
            if mouth_curve < -1.5:
                scores["triste"] += 1.5
            if avg_eye_open > 6.0 and mouth_curve < -2.0:
                scores["triste"] += 1.0
            
            if avg_eyebrow_dist < 4.5:
                scores["brava"] += 6.0
            elif avg_eyebrow_dist < 5.5:
                scores["brava"] += 4.0
            elif avg_eyebrow_dist < 6.5:
                scores["brava"] += 2.0
            
            if eyebrow_drop < -1.0:
                scores["brava"] += 2.0
            
            scores["normal"] = 1.0
            
            max_score = max(scores.values())
            if max_score < 3.0:
                return "normal"
            
            for emotion, score in scores.items():
                if score == max_score:
                    return emotion
            
            return "normal"
            
        except (IndexError, KeyError) as e:
            return "normal"
    
    def _get_stable_emotion(self) -> Optional[str]:
        """Retorna a emoção mais comum no histórico"""
        if len(self._history) == 0:
            return None
        
        counts = Counter(self._history)
        most_common = counts.most_common(1)[0][0]
        
        threshold = max(1, int(len(self._history) * 0.4))
        if counts[most_common] >= threshold:
            return most_common
        
        return most_common
    
    def close(self):
        """Libera recursos"""
        try:
            if self._face_mesh is not None:
                self._face_mesh.close()
        except Exception:
            pass
