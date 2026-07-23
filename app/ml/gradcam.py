from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, class_idx].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        return cam / (cam.max() + 1e-8)


def save_heatmap_overlay(original: Image.Image, cam: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cam_image = Image.fromarray(np.uint8(255 * cam)).resize(original.size)
    heat = np.array(cam_image)
    base = np.array(original.convert("RGB")).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0] = heat
    overlay[..., 1] = np.clip(255 - heat, 0, 255) * 0.35
    blended = np.clip(base * 0.62 + overlay * 0.38, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(output_path)

