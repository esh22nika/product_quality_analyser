import shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

class ArtifactManager:
    """Safely saves and cleans up inspection heatmaps, overlays, and video artifacts."""

    def __init__(self, artifacts_dir: str | Path = "./artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_image_artifact(
        self,
        inspection_id: str,
        image_name: str,
        image_rgb: np.ndarray
    ) -> Path:
        """Saves an image artifact (RGB numpy array) to disk."""
        target_dir = self.artifacts_dir / inspection_id
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / image_name

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path

    def cleanup_inspection(self, inspection_id: str) -> None:
        """Removes saved artifacts for a specific inspection."""
        target_dir = self.artifacts_dir / inspection_id
        if not inspection_id.startswith(("INSP_", "VID_")) or Path(inspection_id).name != inspection_id:
            raise ValueError("Only inspection IDs can be removed")
        if not target_dir.resolve().is_relative_to(self.artifacts_dir.resolve()):
            raise ValueError("Cleanup target escapes artifact directory")
        if target_dir.exists():
            shutil.rmtree(target_dir)

    def cleanup_all(self) -> None:
        """Removes all stored artifact directories."""
        for item in self.artifacts_dir.iterdir():
            if item.is_dir() and item.name.startswith(("INSP_", "VID_")) and not item.is_symlink():
                shutil.rmtree(item)
