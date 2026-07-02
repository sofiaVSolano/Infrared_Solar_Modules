"""Backbone ResNet18 preentrenado en ImageNet, adaptado a 12 clases."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from config import CLASSES


def build_model(freeze_backbone: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # layer3 + layer4: con ~14k imágenes de train, congelar solo layer4 se quedó
        # corto de capacidad para separar clases visualmente parecidas (Cell,
        # Cell-Multi, Cracking, Vegetation salían muy confundidas entre sí).
        for layer in (model.layer3, model.layer4):
            for param in layer.parameters():
                param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    return model
