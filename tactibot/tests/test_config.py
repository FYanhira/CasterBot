from pathlib import Path

from tactibot.config.loader import REPO_ROOT, load_config


def test_load_example_config():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert "video_path" in cfg
    assert "sam3" in cfg
