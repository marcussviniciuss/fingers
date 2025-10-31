import cv2

from fingers.camera import CameraStream
from fingers.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, DISPLAY_SCALE, FLIP_HORIZONTAL, FULLSCREEN
from fingers.drawer import draw_hands_and_overlays
from fingers.hand_detector import HandDetector
from fingers.finger_counter import FingerCounter


def main() -> None:
    camera_stream = CameraStream(camera_index=CAMERA_INDEX)
    detector = HandDetector()
    counter = FingerCounter(history_size=5)

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

            output_frame = draw_hands_and_overlays(
                frame=frame,
                hand_results=hand_results,
                per_hand_counts=per_hand_counts,
                total_count=total_count,
            )

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
        camera_stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
