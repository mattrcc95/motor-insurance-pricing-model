"""Motor insurance pricing on freMTPL2: claim frequency & severity modelling."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OPENML_CACHE = DATA_DIR / "openml_cache"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "OPENML_CACHE",
    "FIGURES_DIR",
    "RANDOM_STATE",
]
