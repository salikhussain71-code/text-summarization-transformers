"""ROUGE evaluation (Google's ``rouge-score``) with bootstrap confidence intervals.

Scores are F-measures x 100. ROUGE-Lsum is the summary-level variant used in the summarisation
literature: both texts are split into sentences (one per line) before scoring.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from . import config
from .preprocessing import split_sentences, word_count

logging.getLogger("absl").setLevel(logging.WARNING)  # rouge-score logs "Using default tokenizer" on every call

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}


def _as_lines(text: str) -> str:
    """Sentences on separate lines (needed by ROUGE-Lsum)."""
    return "\n".join(split_sentences(text))


def score_pairs(predictions: list[str], references: list[str], use_stemmer: bool = True) -> pd.DataFrame:
    """Per-example precision / recall / F1 for every ROUGE type."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=use_stemmer)
    rows = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(_as_lines(ref), _as_lines(pred))
        row = {}
        for t in ROUGE_TYPES:
            row[f"{t}_p"], row[f"{t}_r"], row[f"{t}_f"] = (scores[t].precision * 100, scores[t].recall * 100,
                                                          scores[t].fmeasure * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        m = float(values.mean()) if len(values) else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def aggregate(per_example: pd.DataFrame, n_boot: int = 1000, seed: int = config.RANDOM_STATE) -> dict:
    """Mean F1 (and P / R) per ROUGE type, with a 95 % bootstrap CI for F1."""
    out: dict[str, float] = {}
    for t in ROUGE_TYPES:
        f = per_example[f"{t}_f"].to_numpy()
        lo, hi = bootstrap_ci(f, n_boot, seed=seed)
        out[t] = float(f.mean())
        out[f"{t}_ci_low"], out[f"{t}_ci_high"] = lo, hi
        out[f"{t}_precision"] = float(per_example[f"{t}_p"].mean())
        out[f"{t}_recall"] = float(per_example[f"{t}_r"].mean())
    return out


def compute_rouge(predictions: list[str], references: list[str], use_stemmer: bool = True,
                  n_boot: int = 1000) -> dict:
    """Aggregate ROUGE scores for a system."""
    return aggregate(score_pairs(predictions, references, use_stemmer), n_boot)


# ------------------------------------------------------------------------------ summary statistics
_TOKEN = re.compile(r"[a-z0-9]+")


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    toks = _TOKEN.findall(text.lower())
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def novel_ngram_ratio(summary: str, article: str, n: int = 2) -> float:
    """Share of the summary's n-grams that do **not** occur in the article (0 = pure copy)."""
    grams = _ngrams(summary, n)
    if not grams:
        return 0.0
    source = set(_ngrams(article, n))
    return sum(g not in source for g in grams) / len(grams)


def summary_stats(predictions: list[str], articles: list[str], references: list[str]) -> dict:
    """Length, compression and abstractiveness statistics."""
    pw = np.array([word_count(p) for p in predictions], dtype=float)
    aw = np.array([word_count(a) for a in articles], dtype=float)
    rw = np.array([word_count(r) for r in references], dtype=float)
    return {
        "summary_words": float(pw.mean()),
        "reference_words": float(rw.mean()),
        "compression": float(np.mean(pw / np.maximum(aw, 1))),
        "novel_bigrams": float(np.mean([novel_ngram_ratio(p, a, 2) for p, a in zip(predictions, articles)])),
        "novel_unigrams": float(np.mean([novel_ngram_ratio(p, a, 1) for p, a in zip(predictions, articles)])),
    }
