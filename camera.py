"""Camera capture thread."""
import cv2, time
from config import CAMERA_URL
from state import latest_frame, frame_lock
from tts import speak


def camera_thread():
    while True:
        cap = cv2.VideoCapture(CAMERA_URL)
        if not cap.isOpened():
            speak("Camera could not be opened. Retrying in 3 seconds.")
            time.sleep(3)
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                speak("Failed to grab frame from camera. Reconnecting...")
                cap.release()
                time.sleep(2)
                break
            with frame_lock:
                latest_frame[0] = frame
        cap.release()
