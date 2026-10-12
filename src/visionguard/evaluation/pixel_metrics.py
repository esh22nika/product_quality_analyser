from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class PixelMetricsCalculator:
    """Calculates pixel-level AUROC, pixel average precision, Dice, and IoU."""

    @staticmethod
    def calculate(
        gt_masks: List[np.ndarray],
        anomaly_maps: List[np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Calculates pixel-wise localization metrics across test images."""
        if not gt_masks or not anomaly_maps or len(gt_masks) != len(anomaly_maps):
            return {"error": "Invalid or mismatched mask inputs."}

        flat_gt = []
        flat_maps = []

        for gt, m in zip(gt_masks, anomaly_maps):
            # Ensure same dimensions
            if gt.shape != m.shape:
                import cv2
                m = cv2.resize(m, (gt.shape[1], gt.shape[0]))
            
            # Normalize GT to binary 0 or 1
            gt_bin = (gt > 0).astype(np.uint8)
            flat_gt.append(gt_bin.ravel())
            flat_maps.append(m.ravel())

        all_gt = np.concatenate(flat_gt)
        all_maps = np.concatenate(flat_maps)

        if len(np.unique(all_gt)) < 2:
            return {
                "error": "Pixel ground truth must contain both defect and normal pixels.",
                "pixel_auroc": None
            }

        pixel_auroc = float(roc_auc_score(all_gt, all_maps))
        
        # Subsample for pixel AP if data is very large
        sample_size = min(len(all_gt), 500000)
        idx = np.random.choice(len(all_gt), sample_size, replace=False)
        pixel_ap = float(average_precision_score(all_gt[idx], all_maps[idx]))

        # Calculate IoU and Dice at threshold
        pred_bin = (all_gt >= threshold).astype(np.uint8)
        intersection = np.sum((pred_bin == 1) & (all_gt == 1))
        union = np.sum((pred_bin == 1) | (all_gt == 1))
        iou = float(intersection / union) if union > 0 else 0.0
        dice = float((2 * intersection) / (np.sum(pred_bin) + np.sum(all_gt))) if (np.sum(pred_bin) + np.sum(all_gt)) > 0 else 0.0

        return {
            "pixel_auroc": round(pixel_auroc, 4),
            "pixel_average_precision": round(pixel_ap, 4),
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "threshold": threshold
        }
