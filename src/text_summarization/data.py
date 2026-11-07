"""Dataset loading for the CNN / DailyMail summarisation data (or any article -> summary CSV).

The Kaggle CNN-DailyMail files have the columns ``id``, ``article`` and ``highlights``. Articles are
cleaned of agency boilerplate; reference summaries keep one highlight per line.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config
from .preprocessing import clean_article, clean_summary, word_count

logger = logging.getLogger(__name__)


def find_split_files(data_dir: str | Path) -> dict[str, Path]:
    """Locate ``train`` / ``validation`` / ``test`` CSV files inside ``data_dir``."""
    data_dir = Path(data_dir)
    found = {}
    for split, names in config.SPLIT_CANDIDATES.items():
        for name in names:
            if (data_dir / name).exists():
                found[split] = data_dir / name
                break
    return found


def load_dataset(
    path: str | Path,
    text_col: str = config.TEXT_COL,
    summary_col: str = config.SUMMARY_COL,
    id_col: str | None = config.ID_COL,
    clean: bool = True,
    min_article_words: int = 5,
    min_summary_words: int = 2,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Return a frame with columns ``id``, ``article`` and ``reference``.

    Rows with empty / very short text are dropped. ``nrows`` reads only the first rows of huge files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: '{path}'.\n"
            "Download the Kaggle 'CNN-DailyMail News Text Summarization' dataset and place train.csv, "
            f"validation.csv and test.csv in '{config.RAW_DIR}' (see data/README.md), or pass --data / --data-dir."
        )
    raw = pd.read_csv(path, nrows=nrows)
    missing = {text_col, summary_col} - set(raw.columns)
    if missing:
        raise KeyError(
            f"Column(s) {sorted(missing)} not found in {path.name}. Available columns: {list(raw.columns)}. "
            "Use --text-col / --summary-col."
        )
    ids = raw[id_col].astype(str) if id_col and id_col in raw.columns else pd.Series(raw.index.astype(str))
    df = pd.DataFrame({"id": ids, "article": raw[text_col], "reference": raw[summary_col]})
    df = df.dropna(subset=["article", "reference"])
    if clean:
        df["article"] = df["article"].map(clean_article)
        df["reference"] = df["reference"].map(clean_summary)
    else:
        df["article"] = df["article"].astype(str)
        df["reference"] = df["reference"].astype(str)
    n_before = len(df)
    keep = (df["article"].map(word_count) >= min_article_words) & (df["reference"].map(word_count) >= min_summary_words)
    df = df[keep].reset_index(drop=True)
    logger.info("Loaded %s: %d rows kept of %d.", path.name, len(df), n_before)
    if df.empty:
        raise ValueError(f"No usable rows in {path.name} after cleaning.")
    return df


def sample_rows(df: pd.DataFrame, n: int | None, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Reproducible random subset of ``n`` rows (all rows if ``n`` is None or larger than the data)."""
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def dataset_stats(df: pd.DataFrame) -> dict:
    """Word-level statistics used in the README and the overview figure."""
    a = df["article"].map(word_count)
    r = df["reference"].map(word_count)
    return {
        "rows": int(len(df)),
        "article_words_mean": float(a.mean()), "article_words_median": float(a.median()),
        "article_words_p95": float(a.quantile(0.95)),
        "summary_words_mean": float(r.mean()), "summary_words_median": float(r.median()),
        "compression_mean": float((r / a).mean()),
    }
