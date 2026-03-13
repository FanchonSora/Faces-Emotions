# Faces-Emotions

A real-time facial emotion recognition system that connects AI-based emotion detection with interactive behavior in a Unity 3D environment.

The system uses a Vision Transformer (ViT) model to classify facial emotions from camera input. Detected emotions are streamed through a FastAPI WebSocket server to Unity, where they are mapped to behaviors and movements of 3D characters or objects.

This allows emotional feedback from a user to dynamically influence a virtual environment.

---

## Features

- **Emotion Classification**
  - Detects 5 emotions: Angry, Fear, Happy, Sad, Surprise

- **Real-time Emotion Recognition**
  - Webcam-based emotion detection
  - Live inference using PyTorch

- **Unity Interaction**
  - Real-time WebSocket communication
  - Emotion-to-behavior mapping in Unity

- **Multiple Inference Modes**
  - Single image prediction
  - Webcam real-time detection
  - Server streaming for Unity integration

- **Face Detection**
  - OpenCV Haar Cascade
  - MediaPipe Face Detection

- **Prediction Smoothing**
  - Reduces noise in real-time predictions

---

# System Architecture

```
Webcam
   │
   ▼
Face Detection (OpenCV / MediaPipe)
   │
   ▼
Emotion Classification (ViT Model)
   │
   ▼
FastAPI WebSocket Server
   │
   ▼
Unity Client
   │
   ▼
Emotion → Behavior Mapping
   │
   ▼
3D Character / Object Movement
```

The predicted emotion is transmitted to Unity and translated into specific actions such as character movement, animation, or scene reactions.

---

# Project Structure

```
Faces-Emotions/
├── ai/                          # AI inference components
│   ├── inference/               # Inference scripts
│   │   ├── infer_image.py       # Predict emotion from image file
│   │   ├── realtime_emotion.py  # Real-time emotion detection with MediaPipe
│   │   └── webcam.py            # Webcam emotion detection with smoothing
│   │
│   ├── model/                   # Model architecture and weights
│   │   ├── best_vit_model.pth
│   │   ├── load_model.py
│   │   └── vit_model.py
│   │
│   ├── utils/                   # Utility functions
│   │   ├── emotion_map.py
│   │   └── preprocess.py
│   │
│   └── requirements.txt
│
├── server/                      # FastAPI WebSocket server
│   └── app.py
│
├── EmotionScene/                # Unity project
│   ├── Assets/
│   ├── Packages/
│   ├── ProjectSettings/
│   └── UserSettings/
│
└── README.md
```

---

# Requirements

### Python

- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- FastAPI
- Uvicorn

### Unity

- Unity 2021+
- Universal Render Pipeline (URP)

GPU with CUDA support is recommended for faster inference.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/FanchonSora/Faces-Emotions.git
cd Faces-Emotions
```

---

## Install AI Dependencies

```bash
cd ai
pip install -r requirements.txt
```

---

# Usage

## 1️⃣ Single Image Emotion Prediction

```bash
cd ai
python inference/infer_image.py path/to/image.jpg
```

---

## 2️⃣ Webcam Real-time Emotion Detection

```bash
cd ai
python inference/webcam.py
```

Press **Q** to quit.

---

## 3️⃣ Start WebSocket Server

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

WebSocket endpoint:

```
ws://localhost:8000/ws
```

---

## 4️⃣ Run Unity Scene

1. Open **EmotionScene** in Unity Hub
2. Run the scene
3. Unity connects to the WebSocket server
4. Detected emotions trigger character or object behaviors in the scene

Example:

| Emotion | Unity Behavior |
|------|------|
| Happy | Character jumps / environment brightens |
| Sad | Character slows down |
| Angry | Aggressive movement |
| Surprise | Sudden animation |
| Fear | Defensive behavior |

---

# Model Details

Vision Transformer (ViT)

| Parameter | Value |
|------|------|
Input Size | 128×128 grayscale
Patch Size | 16×16
Embedding Dim | 128
Transformer Depth | 6
Attention Heads | 4
Output Classes | 5 emotions

---

# Emotion Mapping

Emotion predictions are converted to Unity-friendly labels.

Example mapping:

```
Angry     → angry
Fear      → fear
Happy     → happy
Sad       → sad
Surprise  → surprise
```

Unity then maps these labels to character behaviors or animations.

---

# Contributing

1. Fork the repository  
2. Create a branch

```
git checkout -b feature/new-feature
```

3. Commit changes

```
git commit -m "Add new feature"
```

4. Push to GitHub

```
git push origin feature/new-feature
```

5. Open a Pull Request

---

# License

MIT License

---

# Acknowledgments

- Vision Transformer based on the paper  
  **"An Image is Worth 16x16 Words"**
- Face detection using **OpenCV** and **MediaPipe**
- Unity integration for **interactive emotional environments**