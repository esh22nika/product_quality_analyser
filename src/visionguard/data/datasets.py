from typing import Any, Dict
import os
from pathlib import Path
from typing import Dict, List, Optional
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.data")

class DatasetManager:
    """Manages paths, dataset configuration, and preparation helpers."""

    def __init__(self, root: str | Path = "./datasets", dataset_name: str = "mvtec_ad"):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.dataset_path = self.root / dataset_name

    def get_category_path(self, category: str) -> Path:
        return self.dataset_path / category

    def list_categories(self) -> List[str]:
        if not self.dataset_path.exists():
            return []
        return [d.name for d in self.dataset_path.iterdir() if d.is_dir()]

    def is_category_ready(self, category: str) -> bool:
        cat_path = self.get_category_path(category)
        train_good = cat_path / "train" / "good"
        test_dir = cat_path / "test"
        return train_good.exists() and test_dir.exists() and any(train_good.iterdir())
