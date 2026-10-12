import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from visionguard.models.loader import ModelManager
from visionguard.data.transforms import ImagePreprocessor
from visionguard.inference.quality_checks import ImageQualityChecker
from visionguard.inference.postprocessing import AnomalyPostprocessor
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.inference.image")

class ImageInspectionPipeline:
    """End-to-end pipeline for single image defect detection and localization."""

    def __init__(
        self,
        model_name: str = "patchcore",
        category: Optional[str] = "bottle",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.category = category
        self.device = device
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = AnomalyPostprocessor()
        self.quality_checker = ImageQualityChecker()

    def inspect(
        self,
        image_input: str | Path | np.ndarray,
        decision_threshold: float = 0.5,
        pixel_threshold: float = 0.5,
        min_region_area: int = 50
    ) -> Dict[str, Any]:
        """Runs validation, quality checks, model prediction, and heatmap generation."""
        # 1. Load and validate input
        rgb_np, _ = self.preprocessor.load_and_validate(image_input)

        # 2. Quality assessment
        quality_result = self.quality_checker.check_quality(rgb_np)

        # 3. Model inference
        model = ModelManager.get_model(
            model_name=self.model_name,
            category=self.category,
            device=self.device
        )
        
        # Override threshold if supplied
        model.threshold = decision_threshold
        pred = model.predict(rgb_np)

        if pred["error"] or pred["anomaly_map"] is None:
            return {
                "success": False,
                "error": pred["error"] or "Unknown prediction error",
                "quality": quality_result
            }

        anomaly_map = pred["anomaly_map"]

        # 4. Postprocessing
        heatmap_rgb = self.postprocessor.generate_heatmap(anomaly_map)
        overlay_rgb = self.postprocessor.generate_overlay(rgb_np, heatmap_rgb)
        mask = self.postprocessor.apply_threshold(anomaly_map, threshold=pixel_threshold)
        region_analysis = self.postprocessor.analyze_regions(
            mask, anomaly_map, min_region_area=min_region_area
        )
        bbox_overlay = self.postprocessor.draw_bounding_boxes(overlay_rgb, region_analysis["regions"])

        return {
            "success": True,
            "original_image": rgb_np,
            "anomaly_map": anomaly_map,
            "heatmap": heatmap_rgb,
            "overlay": overlay_rgb,
            "annotated_overlay": bbox_overlay,
            "thresholded_mask": mask,
            "image_score": pred["image_score"],
            "predicted_label": pred["predicted_label"],
            "decision_threshold": decision_threshold,
            "pixel_threshold": pixel_threshold,
            "inference_ms": pred["inference_ms"],
            "quality": quality_result,
            "region_analysis": region_analysis,
            "model_name": pred["model_name"]
        }
