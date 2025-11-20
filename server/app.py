import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from ai.model.load_model import load_vit_model, get_transform, EMOTION_LABELS 
from ai.utils.emotion_map import UNITY_EMOTION_MAP
from ai.utils.preprocess import crop_face

app = FastAPI()

#  INIT
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

model = load_vit_model(device=device)
transform = get_transform()

print(f"📋 Emotion labels: {EMOTION_LABELS}")
print(f"🗺️ Unity emotion map: {UNITY_EMOTION_MAP}")


#  WEBSOCKET MAIN PIPELINE
@app.websocket("/ws")
async def emotion_ws(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Unity WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                import json
                frame_data = json.loads(data)
                frame_list = frame_data["frame"]
                frame_np = np.array(frame_list, dtype=np.uint8).reshape((480, 640, 3))

            except Exception as e:
                print(f"❌ Error decoding frame: {e}")
                await websocket.send_json({"emotion": "neutral"})
                continue

            # Convert RGB -> BGR (OpenCV expects BGR)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Crop face
            face = crop_face(frame_np)
            if face is None:
                print("⚠️ No face detected")
                await websocket.send_json({"emotion": "neutral"})
                continue

            # Transform
            from PIL import Image
            img_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
            emotion_label = EMOTION_LABELS[pred]
            unity_emotion = UNITY_EMOTION_MAP.get(emotion_label, "neutral")

            # Send to Unity
            await websocket.send_json({"emotion": unity_emotion})
            print(f"🤖 Predicted: {emotion_label} → Unity: {unity_emotion}")

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 Unity WebSocket disconnected")