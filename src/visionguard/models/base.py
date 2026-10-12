from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import torch

class AnomalyDetector(ABC):
    """Abstract base class for all VisionGuard anomaly detection models."""

    @abstractmethod
    def train(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Trains or builds feature representations from normal samples."""
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Loads a model weights / feature bank checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray | str) -> Dict[str, Any]:
        """Runs anomaly detection and returns normalized dictionary:
        {
            "model_name": str,
            "image_score": float,
            "anomaly_map": np.ndarray (H, W float32 0.0-1.0),
            "predicted_label": str ("LIKELY_NORMAL" | "REVIEW_REQUIRED" | "LIKELY_ANOMALOUS"),
            "threshold": float,
            "inference_ms": float,
            "error": Optional[str]
        }
        """
        raise NotImplementedError
