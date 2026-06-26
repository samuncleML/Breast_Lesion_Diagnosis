import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


CLASS_NAMES = ["benign", "malignant", "normal"]


class BUSIMobilenet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        base_model = models.mobilenet_v3_small(weights=None)
        self.model = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CombinedBUSIModel(nn.Module):
    def __init__(
        self,
        classification_checkpoint: Optional[str] = None,
        segmentation_model_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification_model = BUSIMobilenet(num_classes=3)
        if classification_checkpoint and os.path.exists(classification_checkpoint):
            state = torch.load(classification_checkpoint, map_location=self.device)
            self.classification_model.load_state_dict(state)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        self.segmentation_model_dir = segmentation_model_dir or os.path.join(
            os.path.dirname(__file__), "final_busi_segformer_model"
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.segmentation_model_dir, local_files_only=True
        )
        self.segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(
            self.segmentation_model_dir, local_files_only=True
        )
        self.segmentation_model.to(self.device)
        self.segmentation_model.eval()

        self.classification_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, object]:
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_pil(image)

    @torch.no_grad()
    def predict_from_pil(self, image: Image.Image) -> Dict[str, object]:
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if image_np.ndim == 3 else image_np
        pil_rgb = Image.fromarray(image_rgb)

        cls_tensor = self.classification_transform(pil_rgb).unsqueeze(0).to(self.device)
        cls_logits = self.classification_model(cls_tensor)
        cls_pred = int(cls_logits.argmax(dim=1).item())
        cls_label = CLASS_NAMES[cls_pred]

        seg_inputs = self.image_processor(images=pil_rgb, return_tensors="pt")
        seg_inputs = {k: v.to(self.device) for k, v in seg_inputs.items()}
        seg_logits = self.segmentation_model(**seg_inputs).logits[0]
        seg_mask = seg_logits.argmax(dim=0).cpu().numpy()
        tumor_mask = (seg_mask == 1).astype(np.uint8)

        overlay_image = self._make_overlay(pil_rgb, tumor_mask)
        side_by_side = self._make_side_by_side(pil_rgb, tumor_mask)

        return {
            "class_name": cls_label,
            "class_index": cls_pred,
            "classification_logits": cls_logits.cpu().numpy(),
            "segmentation_mask": tumor_mask,
            "overlay_image": overlay_image,
            "side_by_side_image": side_by_side,
        }

    def _resize_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask

    def _make_overlay(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        image_np = np.array(image)
        mask = self._resize_mask(mask, image_np)
        rgb_mask = np.zeros_like(image_np)
        rgb_mask[mask > 0] = [255, 0, 0]
        blended = cv2.addWeighted(image_np, 0.75, rgb_mask, 0.35, 0)
        return Image.fromarray(blended)

    def _make_side_by_side(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        image_np = np.array(image)
        mask = self._resize_mask(mask, image_np)
        mask_rgb = np.zeros_like(image_np)
        mask_rgb[mask > 0] = [255, 0, 0]
        combined = np.concatenate([image_np, mask_rgb], axis=1)
        return Image.fromarray(combined)


def run_demo_on_images(
    model: CombinedBUSIModel,
    image_paths: List[str],
    output_dir: str = "combined_outputs",
    device: Optional[torch.device] = None,
) -> List[Dict[str, object]]:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    results: List[Dict[str, object]] = []
    for image_path in image_paths:
        result = model.predict(image_path)
        base_name = Path(image_path).stem
        out_path = Path(output_dir) / f"{base_name}_overlay.png"
        result["overlay_image"].save(out_path)
        results.append({"image_path": image_path, **result})
    return results
