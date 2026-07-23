from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image, ImageOps
from torchvision import transforms


IMAGE_SIZE = 224

preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("L")
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return preprocess(image).unsqueeze(0)

