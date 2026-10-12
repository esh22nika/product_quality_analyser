from typing import Any, Callable, Dict, List
import cv2
import numpy as np

class RobustnessEvaluator:
    """Applies controlled industrial transformations to test inference stability."""

    @staticmethod
    def apply_brightness(image_rgb: np.ndarray, factor: float = 1.2) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def apply_contrast(image_rgb: np.ndarray, factor: float = 0.8) -> np.ndarray:
        mean = np.mean(image_rgb)
        adjusted = np.clip((image_rgb - mean) * factor + mean, 0, 255)
        return adjusted.astype(np.uint8)

    @staticmethod
    def apply_blur(image_rgb: np.ndarray, ksize: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(image_rgb, (ksize, ksize), 0)

    @staticmethod
    def apply_jpeg_compression(image_rgb: np.ndarray, quality: int = 50) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode(".jpg", bgr, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    @staticmethod
    def apply_gaussian_noise(image_rgb: np.ndarray, sigma: float = 15.0) -> np.ndarray:
        noise = np.random.normal(0, sigma, image_rgb.shape)
        noisy = np.clip(image_rgb.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    @classmethod
    def evaluate_perturbation(
        cls,
        image_rgb: np.ndarray,
        predict_fn: Callable[[np.ndarray], Dict[str, Any]],
        perturbation_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Runs inference before and after perturbation and returns stability metrics."""
        base_result = predict_fn(image_rgb)
        
        if perturbation_type == "brightness":
            perturbed = cls.apply_brightness(image_rgb, **kwargs)
        elif perturbation_type == "contrast":
            perturbed = cls.apply_contrast(image_rgb, **kwargs)
        elif perturbation_type == "blur":
            perturbed = cls.apply_blur(image_rgb, **kwargs)
        elif perturbation_type == "jpeg_compression":
            perturbed = cls.apply_jpeg_compression(image_rgb, **kwargs)
        elif perturbation_type == "gaussian_noise":
            perturbed = cls.apply_gaussian_noise(image_rgb, **kwargs)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        perturbed_result = predict_fn(perturbed)

        base_score = base_result["image_score"]
        pert_score = perturbed_result["image_score"]
        score_diff = abs(pert_score - base_score)

        return {
            "perturbation": perturbation_type,
            "params": kwargs,
            "base_score": base_score,
            "perturbed_score": pert_score,
            "absolute_score_difference": round(score_diff, 4),
            "label_preserved": base_result["predicted_label"] == perturbed_result["predicted_label"],
            "base_label": base_result["predicted_label"],
            "perturbed_label": perturbed_result["predicted_label"]
        }
