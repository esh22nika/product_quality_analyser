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
            model_cls = ModelRegistry.get(model_name)
            instance = model_cls(device=device)

            if category:
                ckpt_path = Path(checkpoint_dir) / f"{model_name.lower()}_{category}.pt"
                if ckpt_path.exists():
                    try:
                        instance.load(str(ckpt_path))
                        logger.info(f"Loaded existing checkpoint from {ckpt_path}")
                    except Exception as e:
                        logger.warning(f"Could not load checkpoint {ckpt_path}: {e}")

            cls._instances[key] = instance
        return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._instances.clear()
