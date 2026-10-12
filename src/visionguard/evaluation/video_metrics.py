from typing import Any, Dict, List
import numpy as np
import pandas as pd

class VideoMetricsCalculator:
    """Calculates frame AUROC, event precision/recall, duplicate rate, and latency."""

    @staticmethod
    def calculate(
        frame_gt: List[int],
        frame_scores: List[float],
        detected_events: List[Dict[str, Any]],
        ground_truth_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates video inspection performance metrics."""
        from sklearn.metrics import roc_auc_score

        metrics = {}
        if len(np.unique(frame_gt)) > 1:
            metrics["frame_auroc"] = round(float(roc_auc_score(frame_gt, frame_scores)), 4)
        else:
            metrics["frame_auroc"] = None

        total_gt = len(ground_truth_events)
        total_pred = len(detected_events)

        # Match predicted events to ground truth events by temporal overlap
        matched_gt = set()
        matched_pred = set()

        for p_idx, p_evt in enumerate(detected_events):
            p_start, p_end = p_evt["start_time_s"], p_evt["end_time_s"]
            for g_idx, g_evt in enumerate(ground_truth_events):
                g_start, g_end = g_evt["start_time_s"], g_evt["end_time_s"]
                # Check overlap
                if max(p_start, g_start) <= min(p_end, g_end):
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)

        tp = len(matched_gt)
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        duplicate_alerts = max(0, total_pred - len(matched_gt))
        duplicate_rate = duplicate_alerts / total_pred if total_pred > 0 else 0.0

        metrics.update({
            "event_precision": round(precision, 4),
            "event_recall": round(recall, 4),
            "event_f1": round(f1, 4),
            "total_detected_events": total_pred,
            "total_gt_events": total_gt,
            "duplicate_alert_rate": round(duplicate_rate, 4)
        })

        return metrics
