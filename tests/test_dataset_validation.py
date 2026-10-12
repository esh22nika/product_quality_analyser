from pathlib import Path
from visionguard.data.validation import DatasetValidator

def test_missing_dataset_path(tmp_path):
    fake_path = tmp_path / "non_existent"
    res = DatasetValidator.validate_mvtec_ad(fake_path)
    assert res["valid"] is False
    assert "does not exist" in res["error"]

def test_empty_dataset_structure(tmp_path):
    mvtec_dir = tmp_path / "mvtec"
    mvtec_dir.mkdir()
    res = DatasetValidator.validate_mvtec_ad(mvtec_dir)
    assert res["valid"] is False
