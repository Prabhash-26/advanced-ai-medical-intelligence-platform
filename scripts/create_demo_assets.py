from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]


def create_sample_image() -> None:
    path = ROOT / "artifacts" / "sample_images" / "sample_chest_xray.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (224, 224), 18)
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 38, 108, 184), fill=65, outline=100)
    draw.ellipse((116, 38, 182, 184), fill=64, outline=100)
    draw.rectangle((104, 30, 121, 190), fill=84)
    for box in [(58, 92, 86, 114), (138, 102, 166, 128), (122, 142, 152, 162)]:
        draw.ellipse(box, fill=148)
    image.filter(ImageFilter.GaussianBlur(0.6)).save(path)


def create_demo_model_manifest() -> None:
    model_path = ROOT / "artifacts" / "models" / "medical_cnn.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(
        b"Demo artifact placeholder. Run `python scripts/train_model.py` after installing requirements to generate real PyTorch weights.\n"
    )
    (ROOT / "artifacts" / "models" / "medical_cnn.json").write_text(
        json.dumps(
            {
                "classes": ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"],
                "artifact": "artifacts/models/medical_cnn.pt",
                "status": "placeholder until training script is executed",
                "clinical_use": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    create_sample_image()
    create_demo_model_manifest()

