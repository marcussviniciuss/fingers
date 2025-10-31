import cv2
from pathlib import Path

from fingers.camera import CameraStream
from fingers.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, DISPLAY_SCALE, FLIP_HORIZONTAL, FULLSCREEN, MARGIN_PX
from fingers.drawer import draw_hands_and_overlays, _draw_label
from fingers.hand_detector import HandDetector
from fingers.finger_counter import FingerCounter
from fingers.gesture_detector import detect_gestures, GestureImageDisplay
from fingers.emotion_detector import EmotionDetector


def main() -> None:
    camera_stream = CameraStream(camera_index=CAMERA_INDEX)
    detector = HandDetector()
    counter = FingerCounter(history_size=5)
    emotion_detector = EmotionDetector(history_size=7)
    
    gesture_display = GestureImageDisplay(base_path=Path("."))
    gesture_display.load_images()

    window_name = "Detector de Dedos - Pressione 'q' para sair | 'f' para tela cheia"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    fullscreen = FULLSCREEN
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            frame = camera_stream.read_frame()
            if frame is None:
                break
            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            hand_results = detector.detect_hands(frame)
            per_hand_counts, total_count = counter.update(hand_results)
            
            # left_gesture, right_gesture = detect_gestures(hand_results)
            # overlay_img = gesture_display.update(left_gesture, right_gesture, frame.shape)
            overlay_img = None
            
            emotion, face_bbox = emotion_detector.detect_emotion(frame)

            output_frame = draw_hands_and_overlays(
                frame=frame,
                hand_results=hand_results,
                per_hand_counts=per_hand_counts,
                total_count=total_count,
            )
            
            if emotion and face_bbox:
                x, y, w, h = face_bbox
                
                emotion_colors = {
                    "feliz": (0, 255, 0),      # Verde
                    "triste": (255, 0, 255),   # Magenta
                    "brava": (0, 0, 255),      # Vermelho
                    "normal": (255, 255, 0),   # Ciano
                }
                color = emotion_colors.get(emotion, (255, 255, 255))
                
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                
                corner_radius = 10
                cv2.circle(output_frame, (x, y), corner_radius, color, 2)
                cv2.circle(output_frame, (x + w, y), corner_radius, color, 2)
                cv2.circle(output_frame, (x, y + h), corner_radius, color, 2)
                cv2.circle(output_frame, (x + w, y + h), corner_radius, color, 2)
                
                emotion_text = f"{emotion.upper()}"
                text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = max(25, y - 10)
                
                cv2.rectangle(
                    output_frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    color,
                    -1
                )
                cv2.putText(
                    output_frame,
                    emotion_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            if overlay_img is not None:
                output_frame = gesture_display.draw_on_frame(output_frame, overlay_img)

            display_width = int(CAMERA_WIDTH * DISPLAY_SCALE)
            display_height = int(CAMERA_HEIGHT * DISPLAY_SCALE)
            output_frame = cv2.resize(output_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        detector.close()
        emotion_detector.close()
        camera_stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
