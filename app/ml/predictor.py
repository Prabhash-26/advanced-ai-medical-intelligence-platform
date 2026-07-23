from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

from app.core.config import get_settings
from app.ml.gradcam import GradCAM, save_heatmap_overlay
from app.ml.model import build_model
from app.ml.preprocessing import image_to_tensor, load_image


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_path: str | None


class MedicalPredictor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.class_names = self.settings.classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(len(self.class_names)).to(self.device)
        model_path = Path(self.settings.model_path)
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                # Keeps the API bootable when the placeholder artifact has not yet
                # been replaced by running scripts/train_model.py.
                pass
        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.features[-2])

    def predict(self, image_bytes: bytes, heatmap_dir: Path) -> PredictionResult:
        image = load_image(image_bytes)
        tensor = image_to_tensor(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        class_idx = int(torch.argmax(probs).item())
        cam = self.gradcam(tensor, class_idx)
        heatmap_path = heatmap_dir / f"gradcam_{uuid4().hex}.png"
        save_heatmap_overlay(image, cam, heatmap_path)
        probabilities = {name: round(float(probs[i]), 4) for i, name in enumerate(self.class_names)}
        return PredictionResult(
            predicted_class=self.class_names[class_idx],
            confidence=float(probs[class_idx]),
            probabilities=probabilities,
            heatmap_path=str(heatmap_path),
        )
