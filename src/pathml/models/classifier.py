"""Binary classification model for PCam patches."""
import torch.nn as nn
import timm


def build_classifier(model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Build a timm backbone with a 2-class (tumor / no tumor) head."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=2)
