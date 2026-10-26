from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

class AnomalyPostprocessor:
    """Generates heatmaps, overlays, thresholded masks, and connected-region metrics."""

    @staticmethod
    def generate_heatmap(anomaly_map: np.ndarray, colormap: int = cv2.COLORMAP_JET, scale: str = "calibrated") -> np.ndarray:
        """Render fixed calibrated values or explicitly relative display contrast.

        Relative scaling changes colors only; it must never feed scores, masks or metrics.
        Constant maps stay at the low end instead of manufacturing a hotspot.
        """
        values = np.asarray(anomaly_map, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("A heatmap requires a finite 2D score map")
        if scale == "relative":
            low, high = np.percentile(values, [1, 99])
            if high - low <= 1e-12:
                low, high = float(values.min()), float(values.max())
            norm_map = np.clip((values - low) / (high - low), 0, 1) if high - low > 1e-12 else np.zeros_like(values)
        elif scale == "calibrated":
            norm_map = np.clip(values, 0.0, 1.0)
        else:
            raise ValueError("Heatmap scale must be relative or calibrated")
        uint8_map = (norm_map * 255.0).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(uint8_map, colormap)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def generate_overlay(
        original_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Blends original image with heatmap."""
        if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
            heatmap_rgb = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
        overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return overlay

    @staticmethod
    def apply_threshold(anomaly_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Thresholds anomaly map into binary mask (uint8 0 or 255)."""
        binary = (anomaly_map >= threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def analyze_regions(
        binary_mask: np.ndarray,
        anomaly_map: np.ndarray,
        min_region_area: int = 50
    ) -> Dict[str, Any]:
        """Calculates connected components, bounding boxes, area, and estimated suspicious %."""
        h, w = binary_mask.shape[:2]
        total_pixels = h * w

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        regions: List[Dict[str, Any]] = []
        total_suspicious_pixels = 0

        # Label 0 is background
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_region_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]

            region_mask = (labels == i)
            region_scores = anomaly_map[region_mask]
            peak_score = float(np.max(region_scores)) if len(region_scores) > 0 else 0.0
            mean_score = float(np.mean(region_scores)) if len(region_scores) > 0 else 0.0

            total_suspicious_pixels += area

            regions.append({
                "region_id": i,
                "area_pixels": int(area),
                "bbox": [x, y, bw, bh],
                "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "peak_score": round(peak_score, 4),
                "mean_score": round(mean_score, 4)
            })

        percent_area = (total_suspicious_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        return {
            "num_regions": len(regions),
            "total_suspicious_pixels": int(total_suspicious_pixels),
            "estimated_anomalous_area_percent": round(percent_area, 3),
            "regions": regions
        }

    @staticmethod
    def draw_bounding_boxes(image_rgb: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and labels around detected suspicious regions."""
        annotated = image_rgb.copy()
        for r in regions:
            x, y, w, h = r["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"Peak: {r['peak_score']:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return annotated
