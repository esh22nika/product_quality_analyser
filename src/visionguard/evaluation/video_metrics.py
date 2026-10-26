"""Only report metrics with explicit covered ground truth; one-to-one event matching."""
import numpy as np
from sklearn.metrics import roc_auc_score

class VideoMetricsCalculator:
    @staticmethod
    def calculate(frame_gt=None, frame_scores=None, detected_events=None, ground_truth_events=None, iou_threshold=0.1):
        metrics = dict(evaluation_mode="qualitative", frame_auroc=None, event_precision=None,
                       event_recall=None, duplicate_alert_rate=None)
        if frame_gt is not None:
            labels, scores = np.asarray(frame_gt), np.asarray(frame_scores, dtype=float)
            if len(labels) != len(scores) or not len(labels) or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
                raise ValueError("Frame ground truth and scores must be finite, binary and aligned.")
            metrics["evaluation_mode"] = "provided_ground_truth"
            metrics["frame_auroc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
        if ground_truth_events is None:
            return metrics
        predicted, truth = detected_events or [], ground_truth_events
        candidates, overlaps = [], set()
        for p, pred in enumerate(predicted):
            for g, gt in enumerate(truth):
                intersection = max(0, min(pred["end_time_s"], gt["end_time_s"]) - max(pred["start_time_s"], gt["start_time_s"]))
                union = max(pred["end_time_s"], gt["end_time_s"]) - min(pred["start_time_s"], gt["start_time_s"])
                if intersection > 0:
                    overlaps.add(p)
                if union > 0 and intersection / union >= iou_threshold:
                    candidates.append((intersection / union, p, g))
        used_p, used_g, delays = set(), set(), []
        for _, p, g in sorted(candidates, reverse=True):
            if p not in used_p and g not in used_g:
                used_p.add(p)
                used_g.add(g)
                delays.append(max(0, predicted[p].get("detected_time_s", predicted[p]["start_time_s"]) - truth[g]["start_time_s"]))
        tp = len(used_p)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(truth) if truth else None
        metrics.update(evaluation_mode="provided_ground_truth", event_precision=precision, event_recall=recall,
                       event_f1=2 * precision * recall / (precision + recall) if recall is not None and precision + recall else None,
                       total_detected_events=len(predicted), total_gt_events=len(truth),
                       duplicate_alert_rate=len(overlaps - used_p) / len(predicted) if predicted else 0.0,
                       unmatched_alerts=len(predicted) - tp, mean_detection_delay_s=float(np.mean(delays)) if delays else None,
                       matching="one-to-one temporal IoU", temporal_iou_threshold=iou_threshold)
        return metrics
