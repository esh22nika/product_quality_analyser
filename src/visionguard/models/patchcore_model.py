import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import cv2

from visionguard.models.base import AnomalyDetector
from visionguard.models.registry import ModelRegistry
from visionguard.data.transforms import ImagePreprocessor
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.models.patchcore")

try:
    from anomalib.models import Patchcore as AnomalibPatchcore
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    logger.warning("Anomalib could not be imported. Running in fallback mode.")

@ModelRegistry.register("patchcore")
class PatchCoreModel(AnomalyDetector):
    """PatchCore Anomaly Detection architecture using Anomalib / PyTorch."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        threshold: float = 0.5,
        review_low: float = 0.35,
        review_high: float = 0.65,
        device: str = "auto"
    ):
        self.backbone = backbone
        self.threshold = threshold
        self.review_low = review_low
        self.review_high = review_high
        self.device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.engine = None
        self.is_loaded = False

        if ANOMALIB_AVAILABLE:
            try:
                self.model = AnomalibPatchcore(backbone=self.backbone)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Could not initialize Anomalib Patchcore: {e}")

    def train(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Trains PatchCore memory bank on normal samples."""
        if not ANOMALIB_AVAILABLE:
            raise RuntimeError("Anomalib is required for training PatchCore.")

        root = dataset_config.get("root", "./datasets/mvtec_ad")
        category = dataset_config.get("category", "bottle")
        checkpoint_dir = dataset_config.get("checkpoint_dir", "./checkpoints")
        
        logger.info(f"Training PatchCore on category: {category} from {root}...")
        datamodule = MVTecAD(root=root, category=category, train_batch_size=8, num_workers=0)
        
        engine = Engine(
            default_root_dir=checkpoint_dir,
            accelerator=self.device,
            devices=1,
            max_epochs=1
        )
        
        start_time = time.time()
        engine.fit(datamodule=datamodule, model=self.model)
        train_duration = time.time() - start_time

        ckpt_path = Path(checkpoint_dir) / f"patchcore_{category}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        self.is_loaded = True

        return {
            "status": "success",
            "model_name": "PatchCore",
            "category": category,
            "training_time_s": train_duration,
            "checkpoint_path": str(ckpt_path)
        }

    def load(self, checkpoint_path: str) -> None:
        """Loads trained weights or memory bank."""
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        logger.info(f"Loading PatchCore checkpoint from {ckpt}...")
        state_dict = torch.load(ckpt, map_location=self.device)
        if self.model is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
        self.is_loaded = True

    def predict(self, image: np.ndarray | str) -> Dict[str, Any]:
        """Performs anomaly inference and returns normalized output."""
        start_time = time.time()
        try:
            rgb_np, pil_img = self.preprocessor.load_and_validate(image)
            tensor = self.preprocessor.to_tensor(pil_img).to(self.device)

            h, w = rgb_np.shape[:2]

            if self.model is not None:
                with torch.no_grad():
                    outputs = self.model(tensor)
                    
                    # Extract score and anomaly map
                    if isinstance(outputs, dict):
                        score = float(outputs.get("pred_score", outputs.get("anomaly_score", 0.0)))
                        raw_map = outputs.get("anomaly_map", outputs.get("pred_mask", None))
                    elif hasattr(outputs, "pred_score"):
                        score = float(outputs.pred_score.cpu().item())
                        raw_map = outputs.anomaly_map if hasattr(outputs, "anomaly_map") else None
                    else:
                        score = float(outputs[0]) if len(outputs) > 0 else 0.0
                        raw_map = outputs[1] if len(outputs) > 1 else None

                    if raw_map is not None:
                        if isinstance(raw_map, torch.Tensor):
                            raw_map = raw_map.squeeze().cpu().numpy()
                        # Resize anomaly map back to original input image size
                        anomaly_map = cv2.resize(raw_map.astype(np.float32), (w, h))
                        # Normalize 0..1 safely
                        min_v, max_v = anomaly_map.min(), anomaly_map.max()
                        if max_v > min_v:
                            anomaly_map = (anomaly_map - min_v) / (max_v - min_v)
                        else:
                            anomaly_map = np.zeros((h, w), dtype=np.float32)
                    else:
                        anomaly_map = np.zeros((h, w), dtype=np.float32)
            else:
                # Fallback if uninitialized
                score = 0.0
                anomaly_map = np.zeros((h, w), dtype=np.float32)

            # Three-way decision policy
            if score >= self.review_high:
                label = "LIKELY_ANOMALOUS"
            elif score <= self.review_low:
                label = "LIKELY_NORMAL"
            else:
                label = "REVIEW_REQUIRED"

            inference_ms = (time.time() - start_time) * 1000.0

            return {
                "model_name": "PatchCore",
                "image_score": round(score, 4),
                "anomaly_map": anomaly_map,
                "predicted_label": label,
                "threshold": self.threshold,
                "inference_ms": round(inference_ms, 2),
                "error": None
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            inference_ms = (time.time() - start_time) * 1000.0
            return {
                "model_name": "PatchCore",
                "image_score": 0.0,
                "anomaly_map": None,
                "predicted_label": "REVIEW_REQUIRED",
                "threshold": self.threshold,
                "inference_ms": round(inference_ms, 2),
                "error": str(e)
            }
