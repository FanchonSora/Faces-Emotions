import torch
from torchvision import transforms
from ai.model.vit_model import VisionTransformer

# labels đúng với thứ tự dataset ImageFolder
EMOTION_LABELS = ["Angry", "Fear", "Happy", "Sad", "Suprise"]

def load_vit_model(weight_path="ai/model/best_vit_model.pth", device="cuda"):
    model = VisionTransformer(
        img_size=128,
        patch_size=16,
        in_chans=1,
        num_classes=len(EMOTION_LABELS),
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0
    )

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_transform():
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
