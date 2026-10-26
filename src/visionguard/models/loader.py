from pathlib import Path
from typing import Dict, Optional
from visionguard.models.base import AnomalyDetector
from visionguard.models.registry import ModelRegistry
from visionguard.logging_config import setup_logger

# Import implementations to register them
import visionguard.models.patchcore_model
import visionguard.models.efficientad_model

logger = setup_logger("visionguard.models.loader")

class ModelManager:
    """Manages cached instances of anomaly detection models to prevent re-instantiation."""

    _instances: Dict[str, AnomalyDetector] = {}

    @classmethod
    def get_model(
        cls,
        model_name: str,
        category: Optional[str] = None,
        checkpoint_dir: str = "./checkpoints",
        device: str = "auto"
    ) -> AnomalyDetector:
        """Retrieves or creates a cached model instance and loads checkpoint if available."""
        key = f"{model_name.lower()}_{category or 'default'}"
        if key not in cls._instances:
            ckpt_path = None
            if category:
                base = Path(checkpoint_dir)
                candidates = [base / f"{model_name.lower()}_{category}.pt"]
                if model_name.lower() == "efficientad":
                    candidates.append(base / "public_efficientad" / category / "model.ckpt")
                ckpt_path = next((p for p in candidates if p.is_file()), None)
                if ckpt_path is None:
                    raise FileNotFoundError(
                        f"No trained {model_name} checkpoint for '{category}' in {base.resolve()}. "
                        "Add trained category weights before running inspection."
                    )
            model_cls = ModelRegistry.get(model_name)
            kwargs = {"device": device}
            if model_name.lower() == "patchcore" and ckpt_path is not None:
                # The saved category state already contains the feature extractor.
                # Avoid network requests for an unused initial backbone checkpoint.
                kwargs["pre_trained"] = False
            instance = model_cls(**kwargs)

            if ckpt_path is not None:
                instance.load(str(ckpt_path))
                logger.info(f"Loaded existing checkpoint from {ckpt_path}")

            # Keep memory bounded while users switch between benchmark categories.
            if len(cls._instances) >= 2:
                cls._instances.pop(next(iter(cls._instances)))
            cls._instances[key] = instance
        return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._instances.clear()
