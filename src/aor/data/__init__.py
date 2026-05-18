"""Dataset loaders. Each module exposes ``load_chases(raw_dir, params) -> list[Chase]`` and an
``audit(raw_dir) -> dict`` that reports the facts the data contain without fitting anything."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
DERIVED_DIR = ROOT / "data" / "derived"
CONFIG_PATH = ROOT / "configs" / "data.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
