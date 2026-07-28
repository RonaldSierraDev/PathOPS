"""FastAPI inference service for the PCam classifier."""
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from pathml.models.classifier import build_classifier

app = FastAPI(title="PathML Inference API")

CHECKPOINT_PATH = Path("models/pcam_resnet18.pt")
MODEL_NAME = "resnet18"
LABELS = ("no_tumor", "tumor")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(status_code=503, detail=f"no checkpoint found at {CHECKPOINT_PATH}")
        model = build_classifier(MODEL_NAME, pretrained=False)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=_device))
        _model = model.to(_device).eval()
    return _model


def _preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((96, 96))
    array = np.array(image).transpose(2, 0, 1).copy()
    tensor = torch.from_numpy(array).float() / 255.0
    return tensor.unsqueeze(0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    model = _load_model()
    image = Image.open(file.file)
    tensor = _preprocess(image).to(_device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    label_idx = int(probs.argmax())
    return {"label": LABELS[label_idx], "confidence": float(probs[label_idx])}
