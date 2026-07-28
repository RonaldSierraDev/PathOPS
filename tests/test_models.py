import torch

from pathml.models.classifier import build_classifier


def test_classifier_output_shape():
    model = build_classifier("resnet18", pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 2)
