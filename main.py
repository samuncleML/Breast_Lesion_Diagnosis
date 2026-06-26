import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import models
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Breast Ultrasound CADx System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEGFORMER_PATH = "busi-cllassifisification/final_busi_segformer_model"
if not os.path.exists(SEGFORMER_PATH):
    SEGFORMER_PATH = str(BASE_DIR / "final_busi_segformer_model")
segmenter = pipeline(task="image-segmentation", model=SEGFORMER_PATH)


class BUSIMobilenet(nn.Module):
    def __init__(self, num_classes: int):
        super(BUSIMobilenet, self).__init__()
        base_model = models.mobilenet_v3_small(weights=None)
        in_features = 576
        self.model = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


classifier_model = BUSIMobilenet(num_classes=3)
CLASSIFIER_WEIGHTS_PATH = "busi-cllassifisification/checkpoints/best_mobilenetv3_busi.pth"
if not os.path.exists(CLASSIFIER_WEIGHTS_PATH):
    CLASSIFIER_WEIGHTS_PATH = str(BASE_DIR / "checkpoints" / "best_mobilenetv3_busi.pth")
classifier_model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=DEVICE))
classifier_model.to(DEVICE)
classifier_model.eval()

CLASSIFIER_CLASSES = {0: "Benign", 1: "Malignant", 2: "Normal"}
classifier_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def blend_mask_to_overlay(pil_image: Image.Image, pil_mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    image_array = np.array(pil_image.convert("RGB"))
    mask_array = np.array(pil_mask.convert("L"))

    if image_array.shape[:2] != mask_array.shape:
        mask_array = cv2.resize(mask_array, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay_color = np.zeros_like(image_array)
    overlay_color[:] = [0, 0, 255]

    output = image_array.copy()
    mask_bool = mask_array > 0
    if np.any(mask_bool):
        output[mask_bool] = cv2.addWeighted(image_array, 1 - alpha, overlay_color, alpha, 0)[mask_bool]

    return Image.fromarray(output)


@app.get("/", response_class=HTMLResponse)
async def read_root() -> HTMLResponse:
    index_path = BASE_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG and JPG files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    input_tensor = classifier_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_idx].item() * 100)

    predicted_label = CLASSIFIER_CLASSES[predicted_idx]

    results = segmenter(image)
    tumor_mask = None
    for entity in results:
        if entity.get("label") == "tumor":
            tumor_mask = entity.get("mask")
            break

    if tumor_mask is None and results:
        tumor_mask = results[0].get("mask")

    if tumor_mask is None:
        raise HTTPException(status_code=500, detail="The segmentation model did not return a mask.")

    overlay_image = blend_mask_to_overlay(image, tumor_mask, alpha=0.45)
    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return JSONResponse(
        {
            "label": predicted_label,
            "confidence": round(confidence, 2),
            "overlay_image_base64": overlay_base64,
            "overlay_mime_type": "image/png",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
