import cv2
import torch

def crop_face(frame):
    """Dùng opencv Haarcascade để cắt mặt đơn giản."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]
    return face
