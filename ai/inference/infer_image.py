import cv2
import torch
import argparse
import os
from ai.model.load_model import load_vit_model, get_transform, EMOTION_LABELS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_vit_model(device=device)
transform = get_transform()


def predict_emotion(image_path: str):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2.imread failed to read image: {image_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_tensor = transform(img_gray).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, dim=1).item()

    return EMOTION_LABELS[pred]


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from an image")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    try:
        label = predict_emotion(args.image)
        print(label)
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
