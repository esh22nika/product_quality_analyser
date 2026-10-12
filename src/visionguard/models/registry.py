from typing import Callable, Dict, Type
from visionguard.models.base import AnomalyDetector

class ModelRegistry:
    """Registry for anomaly detection architectures."""

    _registry: Dict[str, Type[AnomalyDetector]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[AnomalyDetector]], Type[AnomalyDetector]]:
        def decorator(subclass: Type[AnomalyDetector]) -> Type[AnomalyDetector]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[AnomalyDetector]:
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[key]

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._registry.keys())
