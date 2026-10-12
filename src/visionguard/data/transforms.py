from pathlib import Path
from typing import Any, Dict
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T

class ImagePreprocessor:
    """Standard image input validator, orientation fixer, and normalizer."""

    def __init__(self, target_size: tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_and_validate(self, image_input: str | Path | np.ndarray | Image.Image) -> tuple[np.ndarray, Image.Image]:
        """Loads and validates an image.
        Returns:
            rgb_np: uint8 RGB numpy array (original dimensions).
            pil_img: PIL RGB image with orientation handled.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image path does not exist: {path}")
            try:
                pil_img = Image.open(path)
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception as e:
                raise ValueError(f"Corrupt or invalid image file: {path}. Error: {e}")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Empty image array provided.")
            if len(image_input.shape) == 2:
                # Grayscale to RGB
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                # RGBA to RGB
                image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        rgb_np = np.array(pil_img)
        return rgb_np, pil_img

    def to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Transforms PIL image to normalized tensor (1, C, H, W)."""
        tensor = self.transform(pil_img)
        return tensor.unsqueeze(0)
