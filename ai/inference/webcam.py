from ai.model.load_model import get_transform, load_vit_model, EMOTION_LABELS
import cv2
import torch
import numpy as np
from collections import deque
from ai.utils.preprocess import crop_face
from ai.utils.emotion_map import UNITY_EMOTION_MAP

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_vit_model(device=device)
transform = get_transform()

cap = cv2.VideoCapture(0)

print("🎥 Webcam ON — press Q to quit")


emotion_history = deque(maxlen=10)

def smooth_emotion(pred):
    emotion_history.append(pred)
    # lấy mode (giá trị xuất hiện nhiều nhất)
    values, counts = np.unique(emotion_history, return_counts=True)
    return values[np.argmax(counts)]

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    face = crop_face(frame)
    if face is None:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            # apply smoothing over last N predictions (mode)
            smoothed_pred = smooth_emotion(pred)
            emotion_label = EMOTION_LABELS[int(smoothed_pred)]
            emotion_unity = UNITY_EMOTION_MAP.get(emotion_label, "neutral")

        cv2.putText(frame, emotion_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
