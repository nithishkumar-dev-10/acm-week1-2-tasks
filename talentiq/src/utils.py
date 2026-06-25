"""
utils.py
Shared utilities: logging setup, path helpers, seed setter.
"""
import logging
import os
import random
from pathlib import Path

import numpy as np
from src.config_loader import load_config

_cfg = load_config()


# Logger
def get_logger(name: str = "talentiq") -> logging.Logger:
    log_path = Path(_cfg["paths"]["logs"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers
        return logger

    logger.setLevel(_cfg["logging"]["level"])
    fmt = logging.Formatter(_cfg["logging"]["format"])

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


#  Reproducibility 
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Path helpers
def get_path(key: str) -> Path:
    """Return an absolute Path from config paths section."""
    root = Path(__file__).resolve().parent.parent
    return root / _cfg["paths"][key]
