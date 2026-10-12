from pathlib import Path
from visionguard.config import Config, load_default_config

def test_load_default_config():
    cfg = load_default_config()
    assert cfg.get("model.name") == "patchcore"
    assert cfg.get("data.dataset") == "mvtec_ad"
    assert "bottle" in cfg.get("data.categories")

def test_config_get_set():
    cfg = Config({"test": {"nested": 123}})
    assert cfg.get("test.nested") == 123
    assert cfg.get("test.unknown", "default") == "default"
    cfg.set("test.new_val", "updated")
    assert cfg.get("test.new_val") == "updated"
