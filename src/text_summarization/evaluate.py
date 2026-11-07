"""Benchmark summarisers on CNN / DailyMail and report ROUGE.

    python -m text_summarization.evaluate --n 100                          # baselines + distilbart + t5-small
    python -m text_summarization.evaluate --models bart-large-cnn t5-base --n 200
    python -m text_summarization.evaluate --models models/t5-small-cnndm   # a model you fine-tuned
    python -m text_summarization.evaluate --models --n 500                 # extractive baselines only (fast)
    python -m text_summarization.evaluate --truncation sentences --long-doc chunk

Every system sees exactly the same sampled articles; ROUGE is computed against the reference highlights
with bootstrap confidence intervals. Predictions, tables, figures and an examples page are saved under
``reports/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import transformers

from . import __version__, config, visualize
from .data import dataset_stats, find_split_files, load_dataset, sample_rows
from .extractive import LeadSummarizer, TextRankSummarizer
from .preprocessing import word_count
from .presets import PRESETS
from .rouge import ROUGE_TYPES, aggregate, score_pairs, summary_stats
from .summarizer import LONG_DOC_MODES, TRUNCATION_MODES, AbstractiveSummarizer, ModelLoadError

logger = logging.getLogger("text_summarization")

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"
BASELINES = ("lead", "textrank", "textrank-lead")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate summarisation systems with ROUGE.")
    p.add_argument("--data-dir", type=Path, default=config.RAW_DIR, help="Folder with train/validation/test CSVs.")
    p.add_argument("--data", type=Path, default=None, help="Explicit CSV file to evaluate on.")
    p.add_argument("--split", choices=["test", "validation", "train"], default="test")
    p.add_argument("--text-col", default=config.TEXT_COL)
    p.add_argument("--summary-col", default=config.SUMMARY_COL)
    p.add_argument("--id-col", default=config.ID_COL)
    p.add_argument("--n", type=int, default=100, help="Number of random articles to evaluate (0 = all).")
    p.add_argument("--models", nargs="*", default=["distilbart-cnn-12-6", "t5-small"],
                   help=f"Abstractive models: preset keys {list(PRESETS)}, any Hugging Face id, or a local folder.")
    p.add_argument("--baselines", nargs="*", default=["lead", "textrank"], choices=BASELINES,
                   help="Extractive baselines to include.")
    p.add_argument("--sentences", type=int, default=3, help="Sentences selected by the extractive baselines.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=None, help="Override the model's input limit.")
    p.add_argument("--truncation", choices=TRUNCATION_MODES, default="tokens",
                   help="tokens: cut at the token limit; sentences: keep whole sentences only.")
    p.add_argument("--long-doc", choices=LONG_DOC_MODES, default="truncate",
                   help="truncate: ignore text beyond the limit; chunk: summarise chunks, then merge.")
    p.add_argument("--num-beams", type=int, default=None)
    p.add_argument("--max-summary-tokens", type=int, default=None)
    p.add_argument("--min-summary-tokens", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for confidence intervals.")
    p.add_argument("--examples", type=int, default=5, help="Number of qualitative examples to write.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-update-readme", action="store_true")
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    return p.parse_args(argv)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(df.columns) - 1)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_baselines(names: list[str], n_sentences: int) -> list:
    out = []
    for name in names:
        if name == "lead":
            out.append(LeadSummarizer(n_sentences))
        elif name == "textrank":
            out.append(TextRankSummarizer(n_sentences))
        elif name == "textrank-lead":
            out.append(TextRankSummarizer(n_sentences, lead_bias=0.3))
    return out


def write_examples(df: pd.DataFrame, predictions: dict[str, list[str]], k: int, path: Path) -> None:
    lines = ["# Qualitative examples", "", "Articles are shortened to 700 characters for readability.", ""]
    for i in range(min(k, len(df))):
        row = df.iloc[i]
        article = row["article"] if len(row["article"]) <= 700 else row["article"][:700].rsplit(" ", 1)[0] + " ..."
        lines += [f"## Example {i + 1} (id `{row['id']}`)", "", "**Article**", "", f"> {article}", "",
                  "**Reference**", ""] + [f"> {ln}" for ln in row["reference"].split("\n")] + [""]
        for name, preds in predictions.items():
            lines += [f"**{name}**", ""] + [f"> {ln}" for ln in (preds[i].split("\n") if preds[i] else ["(empty)"])]
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_readme(body: str) -> bool:
    path = config.README_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return False
    pattern = re.compile(re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.DOTALL)
    path.write_text(pattern.sub(lambda _: f"{RESULTS_START}\n{body}\n{RESULTS_END}", text, count=1), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    # 1. Data -----------------------------------------------------------------------------------
    path = args.data or find_split_files(args.data_dir).get(args.split)
    if path is None:
        logger.error("No '%s' CSV found in '%s'. Download the Kaggle CNN-DailyMail dataset into data/raw/ "
                     "(see data/README.md) or pass --data <file.csv>.", args.split, args.data_dir)
        return 1
    try:
        df = load_dataset(path, args.text_col, args.summary_col, args.id_col,
                          nrows=50_000 if args.split == "train" else None)
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.error("%s", err.args[0] if err.args else err)
        return 1
    df = sample_rows(df, args.n or None, args.seed)
    articles, references = df["article"].tolist(), df["reference"].tolist()
    stats = dataset_stats(df)
    logger.info("Evaluating on %d articles from %s (median %.0f words, reference %.0f words)", len(df),
                Path(path).name, stats["article_words_median"], stats["summary_words_median"])

    # 2. Systems ------------------------------------------------------------------------------------
    systems = build_baselines(args.baselines, args.sentences)
    for name in args.models:
        try:
            systems.append(AbstractiveSummarizer(
                name, device=args.device, max_input_tokens=args.max_input_tokens, truncation=args.truncation,
                long_doc=args.long_doc, num_beams=args.num_beams, max_summary_tokens=args.max_summary_tokens,
                min_summary_tokens=args.min_summary_tokens, dtype=args.dtype))
        except ModelLoadError as err:
            logger.warning("Skipping model '%s': %s", name, err)
    if not systems:
        logger.error("No system to evaluate.")
        return 1

    # 3. Run + score ----------------------------------------------------------------------------------
    rows, predictions, per_example, infos = [], {}, {}, {}
    for system in systems:
        logger.info("Running %s ...", system.name)
        t0 = time.perf_counter()
        preds = system.summarize(articles, batch_size=args.batch_size)
        seconds = time.perf_counter() - t0
        scores = score_pairs(preds, references)
        agg = aggregate(scores, args.bootstrap, args.seed)
        extra = summary_stats(preds, articles, references)
        row = {"system": system.name, "kind": system.kind,
               "parameters_m": round(system.num_parameters_millions, 1) if system.kind == "abstractive" else 0.0,
               "seconds_per_article": seconds / len(df),
               "truncated_share": system.truncation_rate(articles) if system.kind == "abstractive" else 0.0,
               **{t: agg[t] for t in ROUGE_TYPES},
               "rougeL_ci_low": agg["rougeL_ci_low"], "rougeL_ci_high": agg["rougeL_ci_high"], **extra}
        rows.append(row)
        predictions[system.name], per_example[system.name] = preds, scores
        if system.kind == "abstractive":
            infos[system.name] = system.info()
        logger.info("  R1=%.2f R2=%.2f RL=%.2f RLsum=%.2f | %.1f words | %.2fs/article", row["rouge1"],
                    row["rouge2"], row["rougeL"], row["rougeLsum"], row["summary_words"], row["seconds_per_article"])

    table = pd.DataFrame(rows).sort_values("rougeL", ascending=False).reset_index(drop=True)

    # 4. Save -------------------------------------------------------------------------------------------
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.METRICS_DIR / "rouge_results.csv", index=False)
    for name, preds in predictions.items():
        out = pd.DataFrame({"id": df["id"], "reference": references, "prediction": preds})
        out = pd.concat([out, per_example[name][[f"{t}_f" for t in ROUGE_TYPES]].round(2)], axis=1)
        out.to_csv(config.PREDICTIONS_DIR / f"{slug(name)}.csv", index=False)

    shown = pd.DataFrame({
        "System": [f"**{s}**" if i == 0 else s for i, s in enumerate(table["system"])],
        "Type": table["kind"],
        "Params (M)": [f"{v:.0f}" if v else "-" for v in table["parameters_m"]],
        "ROUGE-1": table["rouge1"].map("{:.2f}".format), "ROUGE-2": table["rouge2"].map("{:.2f}".format),
        "ROUGE-L": table["rougeL"].map("{:.2f}".format), "ROUGE-Lsum": table["rougeLsum"].map("{:.2f}".format),
        "ROUGE-L 95% CI": [f"{a:.1f}-{b:.1f}" for a, b in zip(table["rougeL_ci_low"], table["rougeL_ci_high"])],
        "Words": table["summary_words"].map("{:.0f}".format),
        "Novel 2-grams": (table["novel_bigrams"] * 100).map("{:.0f}%".format),
        "s/article": table["seconds_per_article"].map("{:.2f}".format),
    })
    table_md = md_table(shown)
    (config.METRICS_DIR / "rouge_results.md").write_text(table_md + "\n", encoding="utf-8")
    summary = {
        "version": __version__, "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_file": Path(path).name, "split": args.split, "n_articles": len(df), "seed": args.seed,
        "dataset": stats, "models": infos, "truncation": args.truncation, "long_doc": args.long_doc,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "device": next((i["device"] for i in infos.values()), "cpu")},
        "results": json.loads(table.to_json(orient="records")),
    }
    (config.METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_examples(df, predictions, args.examples, config.REPORTS_DIR / "examples.md")

    # 5. Figures --------------------------------------------------------------------------------------------
    figures: list[str] = []
    if not args.no_plots:
        out_dir = config.FIGURES_DIR
        visualize.plot_dataset_overview(df["article"].map(word_count), df["reference"].map(word_count), out_dir)
        visualize.plot_rouge_comparison(table, out_dir)
        lengths = {"Reference": [word_count(r) for r in references]}
        lengths.update({n: [word_count(p) for p in preds] for n, preds in predictions.items()})
        visualize.plot_length_distributions(lengths, out_dir)
        visualize.plot_abstractiveness(table, out_dir)
        visualize.plot_speed_vs_quality(table, out_dir)
        figures = ["rouge_comparison.png", "speed_vs_quality.png", "summary_lengths.png", "abstractiveness.png",
                   "dataset_overview.png"]

    # 6. README ---------------------------------------------------------------------------------------------------
    if not args.no_update_readme:
        parts = [
            f"_Generated automatically on {summary['created']} - **{len(df)}** random articles from "
            f"`{Path(path).name}` ({args.split}), seed {args.seed}, device `{summary['environment']['device']}`, "
            f"truncation `{args.truncation}`, long documents `{args.long_doc}`. ROUGE = F1 x 100, sorted by ROUGE-L._",
            "", table_md, "",
            "`Novel 2-grams` = share of summary bigrams that never appear in the article (0 % = pure copy). "
            "`ROUGE-L 95% CI` is a bootstrap interval over the evaluated articles - with few articles, small "
            "differences between systems are **not** statistically meaningful.",
        ]
        trunc = {n: i for n, i in infos.items()}
        if trunc:
            share = {r["system"]: r["truncated_share"] for r in rows if r["system"] in trunc}
            parts += ["", "Articles longer than the model's input limit: " +
                      ", ".join(f"`{n}` {100 * s:.0f}% (limit {trunc[n]['max_input_tokens']} tokens)"
                                for n, s in share.items()) + "."]
        if figures:
            parts += ["", "| | |", "|:---:|:---:|"]
            cells = [f"![{f}](reports/figures/{f})" for f in figures]
            for i in range(0, len(cells), 2):
                pair = cells[i:i + 2] + [""] * (2 - len(cells[i:i + 2]))
                parts.append("| " + " | ".join(pair) + " |")
        parts += ["", "Side-by-side examples: [`reports/examples.md`](reports/examples.md)."]
        if update_readme("\n".join(parts)):
            logger.info("README.md results section updated.")

    print("\n" + table_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
