from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)

class ImageMetricsCalculator:
    """Calculates AUROC, Average Precision, F1, Recall, and Confusion Matrix."""

    @staticmethod
    def calculate(
        y_true: List[int] | np.ndarray,
        y_scores: List[float] | np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Calculates image-level metrics."""
        y_true = np.array(y_true, dtype=int)
        y_scores = np.array(y_scores, dtype=float)

        if len(np.unique(y_true)) < 2:
            return {
                "error": "At least two distinct classes (normal=0, defect=1) required for metric calculation.",
                "auroc": None,
                "average_precision": None
            }

        auroc = float(roc_auc_score(y_true, y_scores))
        ap = float(average_precision_score(y_true, y_scores))

        y_pred = (y_scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "auroc": round(auroc, 4),
            "average_precision": round(ap, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "threshold": threshold,
            "confusion_matrix": cm,
            "sample_count": len(y_true)
        }
