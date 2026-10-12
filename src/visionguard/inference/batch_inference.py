from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from visionguard.inference.image_inference import ImageInspectionPipeline
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.inference.batch")

class BatchInspectionPipeline:
    """Processes multiple images with fault tolerance for individual corrupt files."""

    def __init__(
        self,
        model_name: str = "patchcore",
        category: Optional[str] = "bottle",
        device: str = "auto"
    ):
        self.pipeline = ImageInspectionPipeline(model_name=model_name, category=category, device=device)

    def inspect_batch(
        self,
        image_paths: List[str | Path],
        decision_threshold: float = 0.5,
        pixel_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Runs batch inspection and returns summarized dataframe and per-image results."""
        results = []
        detailed_records = []

        for idx, img_path in enumerate(image_paths):
            p = Path(img_path)
            try:
                out = self.pipeline.inspect(
                    p,
                    decision_threshold=decision_threshold,
                    pixel_threshold=pixel_threshold
                )
                if out["success"]:
                    record = {
                        "filename": p.name,
                        "status": "SUCCESS",
                        "predicted_label": out["predicted_label"],
                        "image_score": out["image_score"],
                        "suspicious_area_pct": out["region_analysis"]["estimated_anomalous_area_percent"],
                        "num_regions": out["region_analysis"]["num_regions"],
                        "inference_ms": out["inference_ms"],
                        "quality_warning": "; ".join(out["quality"]["warnings"]) if out["quality"]["warnings"] else "None",
                        "error": None
                    }
                    detailed_records.append(out)
                else:
                    record = {
                        "filename": p.name,
                        "status": "FAILED",
                        "predicted_label": "ERROR",
                        "image_score": 0.0,
                        "suspicious_area_pct": 0.0,
                        "num_regions": 0,
                        "inference_ms": 0.0,
                        "quality_warning": "N/A",
                        "error": out.get("error", "Unknown error")
                    }
            except Exception as e:
                logger.warning(f"Failed processing {p.name}: {e}")
                record = {
                    "filename": p.name,
                    "status": "FAILED",
                    "predicted_label": "ERROR",
                    "image_score": 0.0,
                    "suspicious_area_pct": 0.0,
                    "num_regions": 0,
                    "inference_ms": 0.0,
                    "quality_warning": "N/A",
                    "error": str(e)
                }

            results.append(record)

        df = pd.DataFrame(results)

        summary = {
            "total_images": len(results),
            "normal_count": int((df["predicted_label"] == "LIKELY_NORMAL").sum()),
            "review_count": int((df["predicted_label"] == "REVIEW_REQUIRED").sum()),
            "anomalous_count": int((df["predicted_label"] == "LIKELY_ANOMALOUS").sum()),
            "failed_count": int((df["status"] == "FAILED").sum()),
            "avg_inference_ms": float(df[df["status"] == "SUCCESS"]["inference_ms"].mean()) if not df.empty else 0.0
        }

        return {
            "summary": summary,
            "table": df,
            "details": detailed_records
        }
