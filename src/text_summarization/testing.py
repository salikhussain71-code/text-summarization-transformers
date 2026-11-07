"""Build a tiny, randomly initialised seq2seq model that runs fully offline.

Its summaries are gibberish - the point is to exercise the *pipeline* (truncation, batching, ROUGE,
fine-tuning, CLIs) without downloading anything. Used by the unit tests, CI and ``make smoke``.

    python -m text_summarization.testing --family t5 --out models/tiny-t5
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

DEFAULT_CORPUS = [
    "the president said on tuesday that the government would raise taxes next year .",
    "police in london arrested a man after a fire at a school on monday night .",
    "the team won the match 2 - 1 and will play in the final next week .",
    "officials said the new plan will cost ten million dollars and create jobs .",
    "summarize : scientists found a new species of frog in the rain forest .",
    "the company reported higher profits and its shares rose by five percent .",
]


def build_tiny_seq2seq(out_dir: str | Path, family: str = "t5", corpus: list[str] | None = None,
                       seed: int = 0, max_length: int = 128) -> Path:
    """Create and save a tiny tokenizer + model. ``family`` is ``t5`` or ``bart``."""
    if family not in ("t5", "bart"):
        raise ValueError("family must be 't5' or 'bart'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    texts = list(corpus or []) + DEFAULT_CORPUS * 5
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<pad>", "</s>", "<unk>", "<s>"],
        initial_alphabet=list(string.ascii_letters + string.digits + string.punctuation),
    )
    tok.train_from_iterator(texts, trainer)
    eos = tok.token_to_id("</s>")
    if family == "bart":
        bos = tok.token_to_id("<s>")
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", bos), ("</s>", eos)])
    else:
        tok.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", eos)])
    tok.decoder = decoders.BPEDecoder()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
                                   bos_token="<s>", model_max_length=max_length)
    vocab = len(fast)

    if family == "t5":
        config = T5Config(vocab_size=vocab, d_model=32, d_ff=64, d_kv=16, num_layers=2, num_decoder_layers=2,
                          num_heads=2, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        model = T5ForConditionalGeneration(config)
    else:
        config = BartConfig(vocab_size=vocab, d_model=32, encoder_layers=2, decoder_layers=2,
                            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64,
                            decoder_ffn_dim=64, max_position_embeddings=max_length, pad_token_id=0, bos_token_id=3,
                            eos_token_id=1, decoder_start_token_id=1, forced_eos_token_id=1)
        model = BartForConditionalGeneration(config)
    fast.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a tiny random seq2seq model for offline smoke tests.")
    parser.add_argument("--family", choices=["t5", "bart"], default="t5")
    parser.add_argument("--out", type=Path, default=Path("models/tiny-t5"))
    args = parser.parse_args(argv)
    path = build_tiny_seq2seq(args.out, args.family)
    print(f"Tiny {args.family} model written to {path} (random weights - summaries are gibberish).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
