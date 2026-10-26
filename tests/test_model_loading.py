from pathlib import Path

import pytest

from visionguard.models.loader import ModelManager


def test_missing_category_checkpoint_does_not_cache_untrained_model(tmp_path):
    ModelManager.clear_cache()
    with pytest.raises(FileNotFoundError, match="No trained efficientad checkpoint"):
        ModelManager.get_model("efficientad", "bottle", checkpoint_dir=str(tmp_path))
    assert not ModelManager._instances


def test_checkpoint_load_failure_is_not_hidden_or_cached(tmp_path, monkeypatch):
    checkpoint = tmp_path / "efficientad_bottle.pt"
    checkpoint.write_bytes(b"broken")

    class FailingModel:
        def __init__(self, **kwargs):
            pass

        def load(self, path):
            raise ValueError("incompatible checkpoint")

    monkeypatch.setattr("visionguard.models.loader.ModelRegistry.get", lambda name: FailingModel)
    ModelManager.clear_cache()
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        ModelManager.get_model("efficientad", "bottle", checkpoint_dir=str(tmp_path))
    assert not ModelManager._instances


def test_public_lightning_checkpoint_is_resolved(tmp_path, monkeypatch):
    checkpoint = tmp_path / "public_efficientad" / "pcb1" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")

    class RecordingModel:
        def __init__(self, **kwargs):
            self.loaded = None

        def load(self, path):
            self.loaded = Path(path)

    monkeypatch.setattr("visionguard.models.loader.ModelRegistry.get", lambda name: RecordingModel)
    ModelManager.clear_cache()
    try:
        model = ModelManager.get_model("efficientad", "pcb1", checkpoint_dir=str(tmp_path))
        assert model.loaded == checkpoint
    finally:
        ModelManager.clear_cache()
