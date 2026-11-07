"""Central configuration: paths and constants shared across the project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("TS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
README_PATH = PROJECT_ROOT / "README.md"

RANDOM_STATE = 42

# Kaggle "CNN-DailyMail News Text Summarization" column names
TEXT_COL = "article"
SUMMARY_COL = "highlights"
ID_COL = "id"

# Split file names (first match wins)
SPLIT_CANDIDATES = {
    "train": ("train.csv",),
    "validation": ("validation.csv", "valid.csv", "val.csv", "dev.csv"),
    "test": ("test.csv",),
}
