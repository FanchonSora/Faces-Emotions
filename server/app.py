from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import torch
import numpy as np
from ai.model.load_model import load_vit_model, get_transform, EMOTION_LABELS
from ai.utils.emotion_map import UNITY_EMOTION_MAP
from ai.utils.preprocess import crop_face
from PIL import Image

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_vit_model(device=device)
transform = get_transform()

# Unity gửi JSON kiểu:
# { "frame": [...pixel array...] }
class FrameData(BaseModel):
    frame: list


@app.post("/predict")
def predict_emotion(data: FrameData):

    # Convert list → numpy array
    frame_np = np.array(data.frame, dtype=np.uint8)
    frame_np = frame_np.reshape((480, 640, 3))

    # Detect face
    face = crop_face(frame_np)
    if face is None:
        return {"emotion": "neutral"}

    # Convert BGR → RGB
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize về đúng size model
    face_resized = cv2.resize(face_rgb, (224, 224))

    img_pil = Image.fromarray(face_resized)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    emotion_label = EMOTION_LABELS[pred]
    unity_emotion = UNITY_EMOTION_MAP[emotion_label]

    return {"emotion": unity_emotion}
