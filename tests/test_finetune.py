import json

import pandas as pd

from text_summarization import finetune
from text_summarization.summarizer import AbstractiveSummarizer

ARGS = ["--n-train", "48", "--n-val", "10", "--epochs", "3", "--batch-size", "8", "--lr", "5e-3",
        "--max-input-tokens", "96", "--max-target-tokens", "32", "--max-summary-tokens", "16",
        "--min-summary-tokens", "3", "--num-beams", "1"]


def test_finetuning_reduces_loss_and_saves_a_loadable_model(isolated, sample_dir, tiny_t5, tmp_path):
    out = tmp_path / "ft-model"
    code = finetune.main(["--model", tiny_t5, "--data-dir", str(sample_dir), "--output", str(out), *ARGS])
    assert code == 0

    history = pd.read_csv(isolated / "reports/metrics/finetune_history.csv")
    assert list(history["epoch"]) == [0, 1, 2, 3]                       # epoch 0 = before fine-tuning
    assert history["val_loss"].iloc[-1] < history["val_loss"].iloc[0]
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[1]

    summary = json.loads((isolated / "reports/metrics/finetune_summary.json").read_text())
    assert summary["n_train"] == 48 and summary["best_epoch"] in (1, 2, 3)
    assert set(summary["before"]) == set(summary["after"])
    assert (isolated / "reports/figures/finetune_history.png").exists()

    for name in ("model.safetensors", "tokenizer.json", "summarizer_meta.json"):
        assert (out / name).exists()
    reloaded = AbstractiveSummarizer(str(out))
    assert reloaded.prefix == "summarize: " and reloaded.max_input_tokens == 96
    assert len(reloaded.summarize(["The mayor announced a new plan on Monday."])) == 1


def test_collate_masks_padding_and_adds_prefix(tiny_t5):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_t5)
    rows = [("The team won the match 2 - 1 .", "the team won ."),
            ("short .", "the team won the match and will play in the final .")]
    batch = finetune.make_collate(tok, "summarize: ", 32, 16)(rows)
    assert batch["labels"].shape[0] == 2 and (batch["labels"] == -100).any()
    assert int(batch["input_ids"].shape[1]) <= 32 and int(batch["labels"].shape[1]) <= 16
    assert tok.decode(batch["input_ids"][0], skip_special_tokens=True).startswith("summarize")


def test_validation_split_is_carved_out_when_no_validation_file(isolated, sample_dir, tiny_t5, tmp_path):
    code = finetune.main(["--model", tiny_t5, "--train-data", str(sample_dir / "train.csv"), "--data-dir",
                          str(tmp_path), "--output", str(tmp_path / "m"), "--epochs", "1", "--n-train", "30",
                          "--n-val", "6", "--batch-size", "8", "--max-input-tokens", "64", "--num-beams", "1",
                          "--max-summary-tokens", "12", "--min-summary-tokens", "2", "--no-plots"])
    assert code == 0
    summary = json.loads((isolated / "reports/metrics/finetune_summary.json").read_text())
    assert summary["n_val"] == 6 and summary["n_train"] == 24


def test_errors_are_reported_cleanly(isolated, tiny_t5, tmp_path):
    assert finetune.main(["--model", tiny_t5, "--data-dir", str(tmp_path)]) == 1                 # no train data
    assert finetune.main(["--model", str(tmp_path / "nope"), "--data-dir", "data/sample"]) == 1  # no model


def test_gradient_accumulation_runs(isolated, sample_dir, tiny_t5, tmp_path):
    code = finetune.main(["--model", tiny_t5, "--data-dir", str(sample_dir), "--output", str(tmp_path / "ga"),
                          "--n-train", "32", "--n-val", "8", "--epochs", "1", "--batch-size", "4", "--grad-accum", "2",
                          "--max-input-tokens", "64", "--num-beams", "1", "--max-summary-tokens", "10",
                          "--min-summary-tokens", "2", "--no-plots"])
    assert code == 0
