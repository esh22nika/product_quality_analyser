import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"

class Config:
    """Manages VisionGuard configuration with fallback and overrides."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

def load_default_config() -> Config:
    if DEFAULT_CONFIG_PATH.exists():
        return Config.from_yaml(DEFAULT_CONFIG_PATH)
    return Config()
