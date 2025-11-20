import cv2
import mediapipe as mp
import torch
import time
from ai.model.load_model import load_vit_model, get_transform, EMOTION_LABELS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_vit_model(device=device)
transform = get_transform()

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# smoothing
from collections import deque
emotion_queue = deque(maxlen=5)

def smooth_emotion(pred):
    emotion_queue.append(pred)
    return max(set(emotion_queue), key=emotion_queue.count)

def extract_face(frame):
    h, w, _ = frame.shape
    results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None

    det = results.detections[0]
    box = det.location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    wbox = int(box.width * w)
    hbox = int(box.height * h)

    x = max(0, x)
    y = max(0, y)

    return frame[y:y + hbox, x:x + wbox]

def predict_emotion(face):
    import numpy as np
    from PIL import Image

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)

    img = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out).item()

    return EMOTION_LABELS[pred]

def main():
    cap = cv2.VideoCapture(0)
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # face detection
        face = extract_face(frame)
        if face is not None:
            raw_emotion = predict_emotion(face)
            smooth = smooth_emotion(raw_emotion)
            label = f"Emotion: {smooth}"

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        # FPS counter
        now = time.time()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Emotion AI", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
