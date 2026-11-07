"""Fine-tune a pre-trained summarisation model on your own article -> summary data (bonus task).

    python -m text_summarization.finetune --model t5-small --n-train 2000 --epochs 1
    python -m text_summarization.finetune --model t5-small --train-data my_news.csv \\
        --text-col body --summary-col abstract --epochs 3 --output models/t5-my-news
    python -m text_summarization.evaluate --models models/t5-small-finetuned --n 100     # compare afterwards

A plain PyTorch loop (AdamW, linear warm-up / decay, gradient accumulation and clipping). ROUGE on a
held-out validation subset is measured **before** fine-tuning (epoch 0) and after every epoch; the best
epoch is saved together with its tokenizer and a ``summarizer_meta.json`` so ``AbstractiveSummarizer`` can
load it directly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from . import __version__, config, visualize
from .data import find_split_files, load_dataset, sample_rows
from .rouge import compute_rouge
from .summarizer import META_FILE, AbstractiveSummarizer, ModelLoadError

logger = logging.getLogger("text_summarization")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a pre-trained summariser.")
    p.add_argument("--model", default="t5-small", help="Preset key, Hugging Face id or local folder to start from.")
    p.add_argument("--data-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--train-data", type=Path, default=None, help="Training CSV (default: <data-dir>/train.csv).")
    p.add_argument("--val-data", type=Path, default=None, help="Validation CSV (default: <data-dir>/validation.csv).")
    p.add_argument("--text-col", default=config.TEXT_COL)
    p.add_argument("--summary-col", default=config.SUMMARY_COL)
    p.add_argument("--id-col", default=config.ID_COL)
    p.add_argument("--n-train", type=int, default=2000, help="Training articles to use (0 = all).")
    p.add_argument("--n-val", type=int, default=200, help="Validation articles used for loss and ROUGE.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-input-tokens", type=int, default=512, help="Encoder length used in training.")
    p.add_argument("--max-target-tokens", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-summary-tokens", type=int, default=None, help="Generation length for validation ROUGE.")
    p.add_argument("--min-summary-tokens", type=int, default=None)
    p.add_argument("--num-beams", type=int, default=None, help="Beams for validation generation (default: model's).")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast (recommended on modern GPUs).")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to save the model (default: models/<name>-finetuned).")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collate(tokenizer, prefix: str, max_in: int, max_tgt: int):
    """Batch builder: tokenised article (+ prefix) and label ids with padding masked as -100."""
    def collate(batch: list[tuple[str, str]]):
        enc = tokenizer([prefix + a for a, _ in batch], truncation=True, max_length=max_in, padding=True,
                        return_tensors="pt")
        labels = tokenizer(text_target=[r.replace("\n", " ") for _, r in batch], truncation=True,
                           max_length=max_tgt, padding=True, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    return collate


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def validation_loss(model, loader, device, bf16: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with _autocast(device, bf16):
            loss = model(**batch).loss
        n = int((batch["labels"] != -100).sum())
        total += float(loss.detach()) * n
        count += n
    return total / max(count, 1)


def generation_rouge(summarizer: AbstractiveSummarizer, articles: list[str], references: list[str],
                     batch_size: int) -> dict:
    preds = summarizer.summarize(articles, batch_size=batch_size)
    scores = compute_rouge(preds, references, n_boot=100)
    return {"rouge1": scores["rouge1"], "rouge2": scores["rouge2"], "rougeL": scores["rougeL"],
            "rougeLsum": scores["rougeLsum"]}


def save_checkpoint(summarizer: AbstractiveSummarizer, out_dir: Path, base_model: str, max_in: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summarizer.model.save_pretrained(out_dir)
    summarizer.tokenizer.save_pretrained(out_dir)
    meta = {"prefix": summarizer.prefix, "max_input_tokens": max_in, "generation": summarizer.generation,
            "base_model": base_model}
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def finetune(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Run fine-tuning. Returns ``(history, summary)``."""
    set_seed(args.seed)
    files = find_split_files(args.data_dir)
    train_path = args.train_data or files.get("train")
    if train_path is None:
        raise FileNotFoundError(f"No training CSV found in '{args.data_dir}'. Pass --train-data <file.csv>.")
    train_df = load_dataset(train_path, args.text_col, args.summary_col, args.id_col,
                            nrows=50_000)
    val_path = args.val_data or files.get("validation")
    if val_path is not None:
        val_df = load_dataset(val_path, args.text_col, args.summary_col, args.id_col)
        train_df = sample_rows(train_df, args.n_train or None, args.seed)
        val_df = sample_rows(val_df, args.n_val or None, args.seed)
    else:  # no validation file: hold out 10 % of the training articles
        train_df = sample_rows(train_df, args.n_train or None, args.seed)
        n_val = min(args.n_val or len(train_df) // 10, max(1, len(train_df) // 5))
        val_df, train_df = train_df.iloc[:n_val].reset_index(drop=True), train_df.iloc[n_val:].reset_index(drop=True)
    logger.info("Fine-tuning on %d articles, validating on %d.", len(train_df), len(val_df))

    summarizer = AbstractiveSummarizer(
        args.model, device=args.device, max_input_tokens=args.max_input_tokens, num_beams=args.num_beams,
        max_summary_tokens=args.max_summary_tokens, min_summary_tokens=args.min_summary_tokens)
    model, tokenizer, device = summarizer.model, summarizer.tokenizer, summarizer.device
    max_in = summarizer.max_input_tokens
    collate = make_collate(tokenizer, summarizer.prefix, max_in, args.max_target_tokens)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(list(zip(train_df["article"], train_df["reference"])), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(list(zip(val_df["article"], val_df["reference"])), batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collate)

    no_decay = ("bias", "layer_norm", "LayerNorm", "layernorm")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    out_dir = args.output or (config.MODELS_DIR / f"{summarizer.name}-finetuned")
    val_articles, val_refs = val_df["article"].tolist(), val_df["reference"].tolist()
    history: list[dict] = []

    # epoch 0 = the pre-trained model before any fine-tuning
    base_loss = validation_loss(model, val_loader, device, args.bf16)
    base_rouge = generation_rouge(summarizer, val_articles, val_refs, args.eval_batch_size)
    history.append({"epoch": 0, "val_loss": base_loss, **base_rouge})
    logger.info("Before fine-tuning: val loss %.4f | ROUGE-1 %.2f ROUGE-2 %.2f ROUGE-L %.2f", base_loss,
                base_rouge["rouge1"], base_rouge["rouge2"], base_rouge["rougeL"])

    best_key, best_epoch, step_time = (-1.0, float("-inf")), 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device, args.bf16):
                loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            running += float(loss.detach())
            n_batches += 1
            if i % args.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if i % 50 == 0:
                logger.info("  epoch %d  batch %d/%d  loss %.4f", epoch, i, len(train_loader), running / n_batches)
        val_loss = validation_loss(model, val_loader, device, args.bf16)
        rouge = generation_rouge(summarizer, val_articles, val_refs, args.eval_batch_size)
        history.append({"epoch": epoch, "train_loss": running / max(n_batches, 1), "val_loss": val_loss, **rouge})
        logger.info("Epoch %d: train loss %.4f | val loss %.4f | ROUGE-1 %.2f ROUGE-2 %.2f ROUGE-L %.2f", epoch,
                    history[-1]["train_loss"], val_loss, rouge["rouge1"], rouge["rouge2"], rouge["rougeL"])
        key = (rouge["rougeL"], -val_loss)          # best ROUGE-L; ties are broken by lower validation loss
        if key > best_key:
            best_key, best_epoch = key, epoch
            save_checkpoint(summarizer, out_dir, summarizer.source, max_in)
            logger.info("  saved checkpoint of epoch %d -> %s", epoch, out_dir)
    model.eval()

    best = next(h for h in history if h["epoch"] == best_epoch)
    summary = {
        "version": __version__, "base_model": summarizer.source, "output": str(out_dir),
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "epochs": args.epochs,
        "best_epoch": best_epoch, "learning_rate": args.lr, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "seconds": round(time.perf_counter() - step_time, 1),
        "before": {k: base_rouge[k] for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")},
        "after": {k: best[k] for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")},
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.METRICS_DIR / "finetune_history.csv", index=False)
    (config.METRICS_DIR / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        visualize.plot_training_history(history, config.FIGURES_DIR)
    return history, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    try:
        _, summary = finetune(args)
    except (FileNotFoundError, KeyError, ValueError, ModelLoadError) as err:
        logger.error("%s", err.args[0] if err.args else err)
        return 1
    b, a = summary["before"], summary["after"]
    print(f"\nFine-tuned model saved to {summary['output']} (epoch {summary['best_epoch']})")
    print(f"ROUGE-1 {b['rouge1']:.2f} -> {a['rouge1']:.2f} | ROUGE-2 {b['rouge2']:.2f} -> {a['rouge2']:.2f} | "
          f"ROUGE-L {b['rougeL']:.2f} -> {a['rougeL']:.2f}  (validation subset, before -> after)")
    print(f"Evaluate on the test set:  python -m text_summarization.evaluate --models {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
