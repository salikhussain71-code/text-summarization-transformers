"""Summarise your own text: abstractive (pre-trained model), TextRank or Lead-N.

    python -m text_summarization.summarize --model t5-small "Long article text ..."
    python -m text_summarization.summarize --file article.txt --method textrank --sentences 3
    python -m text_summarization.summarize --csv news.csv --text-col content --out summaries.csv
    python -m text_summarization.summarize --model bart-large-cnn --long-doc chunk --file long_report.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .extractive import LeadSummarizer, TextRankSummarizer
from .preprocessing import clean_article, word_count
from .presets import PRESETS
from .summarizer import LONG_DOC_MODES, TRUNCATION_MODES, AbstractiveSummarizer, ModelLoadError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarise text.")
    p.add_argument("text", nargs="*", help="Text(s) to summarise.")
    p.add_argument("--file", type=Path, nargs="*", default=[], help="Text file(s); one article per file.")
    p.add_argument("--csv", type=Path, help="CSV with one article per row.")
    p.add_argument("--text-col", default="article")
    p.add_argument("--out", type=Path, help="Write the summaries to this CSV.")
    p.add_argument("--method", choices=["abstractive", "textrank", "lead"], default="abstractive")
    p.add_argument("--model", default="distilbart-cnn-12-6",
                   help=f"Preset {list(PRESETS)}, Hugging Face id or local folder (abstractive method).")
    p.add_argument("--sentences", type=int, default=3, help="Sentences for textrank / lead.")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--truncation", choices=TRUNCATION_MODES, default="tokens")
    p.add_argument("--long-doc", choices=LONG_DOC_MODES, default="truncate")
    p.add_argument("--num-beams", type=int, default=None)
    p.add_argument("--max-summary-tokens", type=int, default=None)
    p.add_argument("--min-summary-tokens", type=int, default=None)
    p.add_argument("--no-clean", action="store_true", help="Do not strip CNN / Daily Mail boilerplate.")
    args = p.parse_args(argv)

    items: list[tuple[str, str]] = [(f"text {i + 1}", t) for i, t in enumerate(args.text)]
    for f in args.file:
        items.append((f.name, f.read_text(encoding="utf-8")))
    if args.csv:
        frame = pd.read_csv(args.csv)
        if args.text_col not in frame.columns:
            print(f"Column '{args.text_col}' not found. Available: {list(frame.columns)}", file=sys.stderr)
            return 1
        items += [(f"row {i}", str(t)) for i, t in frame[args.text_col].dropna().items()]
    if not items:
        p.print_help()
        return 1

    if args.method == "lead":
        system = LeadSummarizer(args.sentences)
    elif args.method == "textrank":
        system = TextRankSummarizer(args.sentences)
    else:
        try:
            system = AbstractiveSummarizer(
                args.model, device=args.device, max_input_tokens=args.max_input_tokens, truncation=args.truncation,
                long_doc=args.long_doc, num_beams=args.num_beams, max_summary_tokens=args.max_summary_tokens,
                min_summary_tokens=args.min_summary_tokens)
        except ModelLoadError as err:
            print(err, file=sys.stderr)
            return 1

    texts = [t if args.no_clean else clean_article(t) for _, t in items]
    summaries = system.summarize(texts, batch_size=args.batch_size)
    rows = []
    for (name, _), text, summary in zip(items, texts, summaries):
        n_in, n_out = word_count(text), word_count(summary)
        print(f"\n=== {name} ({system.name}) - {n_in} -> {n_out} words ({n_out / max(n_in, 1):.0%}) ===")
        print(summary or "(empty input)")
        rows.append({"source": name, "words_in": n_in, "words_out": n_out, "summary": summary})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved {len(rows)} summaries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
