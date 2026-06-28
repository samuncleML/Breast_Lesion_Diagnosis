import base64
import io
import os
from pathlib import Path
import sys

print(sys.executable)

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

def smooth_mask(mask_array: np.ndarray, blur_kernel: int = 7) -> np.ndarray:
    """Cleans up jagged edges from upscaling using Gaussian Blur and thresholding."""
    if blur_kernel % 2 == 0:
        blur_kernel += 1  # Kernel size must be odd
    blurred = cv2.GaussianBlur(mask_array, (blur_kernel, blur_kernel), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed


def feathered_alpha_mask(mask_array: np.ndarray, feather: int = 9) -> np.ndarray:
    """Creates a soft-edged alpha map instead of a hard cutoff."""
    if feather % 2 == 0:
        feather += 1  # Kernel size must be odd
    # Normalize mask to 0.0 - 1.0
    mask_float = mask_array.astype(np.float32) / 255.0
    # Apply blur to create the soft, feathered edges
    feathered = cv2.GaussianBlur(mask_float, (feather, feather), 0)
    return feathered


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


def blend_mask_to_overlay(
    pil_image: Image.Image,
    pil_mask: Image.Image,
    alpha: float = 0.45,
    draw_contour: bool = True,
) -> Image.Image:
    image_array = np.array(pil_image.convert("RGB"))
    mask_array = np.array(pil_mask.convert("L"))
 
    if image_array.shape[:2] != mask_array.shape:
        # Use INTER_LINEAR here instead of INTER_NEAREST -- gives a
        # smoother base to work with before we even get to smoothing.
        mask_array = cv2.resize(
            mask_array,
            (image_array.shape[1], image_array.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
 
    # 1. Clean up the jagged edges from upscaling
    mask_array = smooth_mask(mask_array, blur_kernel=7)
 
    # 2. Soft-edged alpha instead of hard cutoff
    alpha_map = feathered_alpha_mask(mask_array, feather=9) * alpha  # shape (H, W)
    alpha_map_3ch = np.stack([alpha_map] * 3, axis=-1)  # (H, W, 3)
 
    overlay_color = np.zeros_like(image_array, dtype=np.float32)
    overlay_color[:] = [0, 0, 255]  # red in RGB order matching original
 
    output = (
        image_array.astype(np.float32) * (1 - alpha_map_3ch)
        + overlay_color * alpha_map_3ch
    ).astype(np.uint8)
 
    # 3. Thin anti-aliased contour outline on top (optional, looks clinical)
    if draw_contour:
        contours, _ = cv2.findContours(
            (mask_array > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(output, contours, -1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
 
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

    overlay_image = image if predicted_label == "Normal" else blend_mask_to_overlay(image, tumor_mask, alpha=0.45)
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
