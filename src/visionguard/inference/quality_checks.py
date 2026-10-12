from typing import Any, Dict, List
import cv2
import numpy as np
from PIL import Image

class ImageQualityChecker:
    """Detects unusable, dark, washed-out, or blurry industrial images."""

    @staticmethod
    def check_quality(rgb_image: np.ndarray) -> dict[str, Any]:
        """Analyzes image quality and flags potential issues.
        Returns:
            {
                "is_usable": bool,
                "warnings": list[str],
                "blur_score": float,
                "mean_brightness": float,
                "contrast": float
            }
        """
        warnings = []
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # 1. Blur Detection via Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 30.0:
            warnings.append(f"Image appears blurry (Laplacian variance: {blur_score:.1f} < 30.0)")

        # 2. Brightness check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 25.0:
            warnings.append(f"Inadequate lighting / image too dark (mean brightness: {mean_brightness:.1f})")
        elif mean_brightness > 235.0:
            warnings.append(f"Overexposed or washed out (mean brightness: {mean_brightness:.1f})")

        # 3. Contrast check
        contrast = float(np.std(gray))
        if contrast < 15.0:
            warnings.append(f"Low contrast (std deviation: {contrast:.1f})")

        return {
            "is_usable": len(warnings) == 0,
            "warnings": warnings,
            "blur_score": round(blur_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast": round(contrast, 2)
        }
