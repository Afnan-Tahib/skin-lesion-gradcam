"""EfficientNet-B0 transfer-learning model."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return model


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
