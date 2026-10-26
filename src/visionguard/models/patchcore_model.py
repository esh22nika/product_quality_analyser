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
    from anomalib.data import MVTecAD, Visa
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
        device: str = "auto",
        coreset_sampling_ratio: float = 0.1,
        pre_trained: bool = True
    ):
        self.backbone = backbone
        self.threshold = threshold
        self.review_low = review_low
        self.review_high = review_high
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.engine = None
        self.is_loaded = False

        if ANOMALIB_AVAILABLE:
            try:
                self.model = AnomalibPatchcore(backbone=self.backbone, coreset_sampling_ratio=self.coreset_sampling_ratio,
                                              pre_trained=pre_trained)
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
        dataset_name = dataset_config.get("dataset", "mvtec_ad").lower()
        checkpoint_dir = dataset_config.get("checkpoint_dir", "./checkpoints")
        
        # Determine coreset ratio: VisA has ~1000 high-res images (~1M patch features).
        # On CPU, a ratio of 0.002 (~1800 features) finishes in ~1-2 mins with high AUROC.
        ratio = dataset_config.get("coreset_sampling_ratio")
        if ratio is None:
            ratio = 0.002 if "visa" in dataset_name else 0.1
        
        logger.info(f"Training PatchCore on category: {category} (dataset: {dataset_name}) from {root} with coreset_ratio={ratio}...")
        self.coreset_sampling_ratio = ratio
        self.model = AnomalibPatchcore(backbone=self.backbone, coreset_sampling_ratio=self.coreset_sampling_ratio)
        self.model.to(self.device)

        if "visa" in dataset_name:
            datamodule = Visa(root=root, category=category, train_batch_size=8, num_workers=0,
                              val_split_mode="from_test", val_split_ratio=0.5, seed=42)
        else:
            datamodule = MVTecAD(root=root, category=category, train_batch_size=8, num_workers=0,
                                 val_split_mode="from_test", val_split_ratio=0.5, seed=42)
        
        engine = Engine(
            default_root_dir=checkpoint_dir,
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            max_epochs=1
        )
        
        start_time = time.time()
        self.model.train()
        engine.fit(datamodule=datamodule, model=self.model)
        self.model.eval()
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
        state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
        if self.model is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
        self.checkpoint_path = str(ckpt.resolve())
        self.is_loaded = True

    def predict(self, image):
        from visionguard.models.prediction import predict_checkpoint
        return predict_checkpoint(self, image, "PatchCore")
