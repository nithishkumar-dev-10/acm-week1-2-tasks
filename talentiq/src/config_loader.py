"""
config_loader.py
Loads all YAML config files and exposes them as Python dicts.
"""
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load(filename: str) -> dict:
    with open(CONFIG_DIR / filename, "r") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    return _load("config.yaml")


def load_features() -> dict:
    return _load("features.yaml")


def load_hyperparams() -> dict:
    return _load("hyperparameters.yaml")


# ─── Quick sanity check ─────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    feat = load_features()
    hp = load_hyperparams()
    print(" config.yaml     loaded —", list(cfg.keys()))
    print(" features.yaml   loaded —", list(feat.keys()))
    print(" hyperparams.yaml loaded —", list(hp.keys()))
