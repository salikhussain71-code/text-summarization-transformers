"""Charts. Every function saves a PNG and returns the output path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers / CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#1F4E79", "#4F8FC9", "#2E7D32", "#8BC34A", "#F9A825", "#C62828", "#8E24AA", "#00838F"]
ROUGE_COLORS = {"rouge1": "#1F4E79", "rouge2": "#F9A825", "rougeL": "#2E7D32", "rougeLsum": "#8E24AA"}
ROUGE_LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset_overview(article_words: pd.Series, summary_words: pd.Series, out_dir: Path) -> Path:
    """Article length, reference-summary length and compression ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hi = float(np.percentile(article_words, 99))
    axes[0].hist(article_words.clip(upper=hi), bins=40, color=PALETTE[0])
    axes[0].axvline(article_words.median(), color="k", ls="--", lw=1, label=f"median {article_words.median():.0f}")
    axes[0].set_title("Article length (words)")
    axes[0].legend()
    axes[1].hist(summary_words.clip(upper=float(np.percentile(summary_words, 99))), bins=30, color=PALETTE[2])
    axes[1].axvline(summary_words.median(), color="k", ls="--", lw=1, label=f"median {summary_words.median():.0f}")
    axes[1].set_title("Reference summary length (words)")
    axes[1].legend()
    ratio = (summary_words / article_words.clip(lower=1)).clip(upper=1)
    axes[2].hist(ratio, bins=40, color=PALETTE[4])
    axes[2].axvline(ratio.median(), color="k", ls="--", lw=1, label=f"median {ratio.median():.2f}")
    axes[2].set_title("Compression (summary / article words)")
    axes[2].legend()
    for ax in axes:
        ax.set_ylabel("Articles")
    return _save(fig, out_dir / "dataset_overview.png")


def plot_rouge_comparison(table: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped horizontal bars: four ROUGE variants per system."""
    df = table.sort_values("rougeL")
    y = np.arange(len(df))
    types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    h = 0.8 / len(types)
    fig, ax = plt.subplots(figsize=(10, 0.95 * len(df) + 2))
    for k, t in enumerate(types):
        bars = ax.barh(y + (k - 1.5) * h, df[t], height=h * 0.92, color=ROUGE_COLORS[t], label=ROUGE_LABELS[t])
        ax.bar_label(bars, fmt="{:.1f}", fontsize=7, padding=2)
    ax.set_yticks(y, df["system"])
    ax.set_xlim(0, max(df[types].max()) * 1.12)
    ax.set_xlabel("F1 x 100")
    ax.set_title("ROUGE comparison")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "rouge_comparison.png")


def plot_length_distributions(lengths: dict[str, list[float]], out_dir: Path) -> Path:
    """Word count of the reference and of every system's summaries (box plots)."""
    names = list(lengths)
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 2))
    key = "tick_labels" if "tick_labels" in boxplot_params() else "labels"
    box = ax.boxplot([lengths[n] for n in names], vert=False, patch_artist=True, showfliers=False, **{key: names})
    for patch, color in zip(box["boxes"], PALETTE * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xlabel("Words per summary")
    ax.set_title("Summary length by system")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "summary_lengths.png")


def boxplot_params() -> set[str]:
    """Names of matplotlib's ``boxplot`` parameters (``labels`` was renamed ``tick_labels`` in 3.9)."""
    import inspect

    return set(inspect.signature(plt.Axes.boxplot).parameters)


def plot_abstractiveness(table: pd.DataFrame, out_dir: Path) -> Path:
    """Share of summary bigrams that never occur in the article (0 = copy-paste)."""
    df = table.sort_values("novel_bigrams")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.8))
    bars = ax.barh(df["system"], df["novel_bigrams"] * 100, color=PALETTE[1])
    ax.bar_label(bars, fmt="{:.1f}%", padding=3, fontsize=8)
    ax.set_xlabel("Novel bigrams (%)")
    ax.set_title("Abstractiveness: bigrams not copied from the article")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "abstractiveness.png")


def plot_speed_vs_quality(table: pd.DataFrame, out_dir: Path) -> Path:
    """ROUGE-L against seconds per article."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, row in enumerate(table.itertuples()):
        ax.scatter(max(row.seconds_per_article, 1e-4), row.rougeL, s=90, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(row.system, (max(row.seconds_per_article, 1e-4), row.rougeL), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per article (log scale)")
    ax.set_ylabel("ROUGE-L (F1 x 100)")
    ax.set_title("Speed vs quality")
    return _save(fig, out_dir / "speed_vs_quality.png")


def plot_training_history(history: list[dict], out_dir: Path) -> Path:
    """Fine-tuning curves: losses and validation ROUGE per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    train = [(h["epoch"], h["train_loss"]) for h in history if "train_loss" in h]
    val = [(h["epoch"], h["val_loss"]) for h in history if "val_loss" in h]
    if train:
        axes[0].plot(*zip(*train), "o-", label="train")
    if val:
        axes[0].plot(*zip(*val), "o-", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    for t in ("rouge1", "rouge2", "rougeL"):
        points = [(h["epoch"], h[t]) for h in history if t in h]
        if points:
            axes[1].plot(*zip(*points), "o-", color=ROUGE_COLORS[t], label=ROUGE_LABELS[t])
    axes[1].set_title("Validation ROUGE (epoch 0 = before fine-tuning)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    return _save(fig, out_dir / "finetune_history.png")
