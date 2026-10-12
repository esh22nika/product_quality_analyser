import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

class DatasetValidator:
    """Validates local dataset structures (MVTec AD, VisA) and generates summaries."""

    SUPPORTED_DATASETS = ["mvtec_ad", "visa"]

    @staticmethod
    def validate_mvtec_ad(root: str | Path, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validates MVTec AD directory layout and sample counts."""
        root_path = Path(root)
        if not root_path.exists():
            return {
                "valid": False,
                "error": f"Root directory '{root_path}' does not exist.",
                "categories": {}
            }

        available_categories = [d.name for d in root_path.iterdir() if d.is_dir()]
        target_cats = categories if categories else available_categories

        result = {
            "valid": True,
            "root": str(root_path),
            "total_categories": len(available_categories),
            "categories": {},
            "warnings": []
        }

        if not available_categories:
            result["valid"] = False
            result["error"] = f"No category subdirectories found under {root_path}."
            return result

        for cat in target_cats:
            cat_dir = root_path / cat
            if not cat_dir.exists():
                result["warnings"].append(f"Category '{cat}' not found under {root_path}")
                continue

            train_good = cat_dir / "train" / "good"
            test_dir = cat_dir / "test"
            gt_dir = cat_dir / "ground_truth"

            train_count = len(list(train_good.glob("*.png"))) if train_good.exists() else 0
            
            test_counts = {}
            if test_dir.exists():
                for sub in test_dir.iterdir():
                    if sub.is_dir():
                        test_counts[sub.name] = len(list(sub.glob("*.png")))

            gt_counts = {}
            if gt_dir.exists():
                for sub in gt_dir.iterdir():
                    if sub.is_dir():
                        gt_counts[sub.name] = len(list(sub.glob("*.png")))

            cat_valid = train_count > 0 and len(test_counts) > 0
            if not cat_valid:
                result["warnings"].append(f"Category '{cat}' is incomplete (train={train_count}, test_subdirs={len(test_counts)})")

            result["categories"][cat] = {
                "valid": cat_valid,
                "train_good_count": train_count,
                "test_counts": test_counts,
                "test_total": sum(test_counts.values()),
                "ground_truth_counts": gt_counts,
            }

        return result
