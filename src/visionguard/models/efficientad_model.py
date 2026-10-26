import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor

from visionguard.models.base import AnomalyDetector
from visionguard.models.registry import ModelRegistry
from visionguard.data.transforms import ImagePreprocessor
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.models.efficientad")

try:
    from anomalib.models import EfficientAd as AnomalibEfficientAd
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD, Visa
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    logger.warning("Anomalib could not be imported for EfficientAD.")

@ModelRegistry.register("efficientad")
class EfficientADModel(AnomalyDetector):
    """EfficientAD Student-Teacher anomaly detection model."""

    def __init__(
        self,
        model_size: str = "small",
        threshold: float = 0.5,
        review_low: float = 0.35,
        review_high: float = 0.65,
        device: str = "auto"
    ):
        self.model_size = {"s": "small", "m": "medium"}.get(model_size, model_size)
        self.threshold = threshold
        self.review_low = review_low
        self.review_high = review_high
        self.device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.is_loaded = False

        if ANOMALIB_AVAILABLE:
            try:
                self.model = AnomalibEfficientAd(model_size=self.model_size, visualizer=False)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Could not initialize Anomalib EfficientAD: {e}")

    def train(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Trains EfficientAD student-teacher model."""
        if not ANOMALIB_AVAILABLE:
            raise RuntimeError("Anomalib is required for training EfficientAD.")

        root = dataset_config.get("root", "./datasets/mvtec_ad")
        category = dataset_config.get("category", "bottle")
        checkpoint_dir = dataset_config.get("checkpoint_dir", "./checkpoints")

        logger.info(f"Training EfficientAD on category: {category} from {root}...")
        data_cls = Visa if dataset_config.get("dataset") == "visa" else MVTecAD
        datamodule = data_cls(root=root, category=category, train_batch_size=1, num_workers=0,
                              val_split_mode="from_test", val_split_ratio=0.5, seed=42)
        
        engine = Engine(
            default_root_dir=checkpoint_dir,
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            max_epochs=dataset_config.get("max_epochs", 5)
        )

        start_time = time.time()
        engine.fit(datamodule=datamodule, model=self.model)
        train_duration = time.time() - start_time

        ckpt_path = Path(checkpoint_dir) / f"efficientad_{category}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        self.is_loaded = True

        return {
            "status": "success",
            "model_name": "EfficientAD",
            "category": category,
            "training_time_s": train_duration,
            "checkpoint_path": str(ckpt_path)
        }

    def load(self, checkpoint_path: str) -> None:
        """Loads trained EfficientAD weights."""
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        logger.info(f"Loading EfficientAD checkpoint from {ckpt}...")
        if not ANOMALIB_AVAILABLE:
            raise RuntimeError("Anomalib is required to load EfficientAD checkpoints.")
        # Category checkpoints can contain serialized training configuration. Only
        # accept tensors and library enums; never execute repository-provided code.
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
        with torch.serialization.safe_globals([EfficientAdModelSize]):
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # The medium PDN has six convolutions; the small PDN has four.
        size = "medium" if any("teacher.conv6." in k for k in state_dict) else "small"
        params = checkpoint.get("hyper_parameters", {})
        self.model = AnomalibEfficientAd(
            model_size=size, visualizer=False, evaluator=False,
            teacher_out_channels=int(params.get("teacher_out_channels", 384)),
            padding=bool(params.get("padding", False)), pad_maps=bool(params.get("pad_maps", True)),
        )
        # Evaluation accumulators differ across Anomalib versions and are not part
        # of inference. Keep every network, normalization and calibration key strict.
        inference_state = {k: v for k, v in state_dict.items() if not k.startswith("evaluator.")}
        self.model.load_state_dict(inference_state, strict=True)
        self.model_size = size
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = str(ckpt.resolve())
        self.is_loaded = True

    def predict(self, image):
        from visionguard.models.prediction import predict_checkpoint
        return predict_checkpoint(self, image, "EfficientAD")
