"""Abstractive summarisation with pre-trained encoder-decoder models (BART, T5, Pegasus, ...).

Highlights
----------
* Works with a preset key (``t5-small``), any Hugging Face model id, or a local directory (for example a
  model you fine-tuned with ``text_summarization.finetune``).
* **Model input limits** are respected: ``truncation="tokens"`` cuts at the token limit,
  ``truncation="sentences"`` keeps only whole sentences; ``long_doc="chunk"`` summarises very long
  documents chunk by chunk and then summarises the summaries.
* T5-style task prefixes (``summarize: ``) are added automatically.
* Batches are length-sorted for speed and results are returned in the original order.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .preprocessing import chunk_sentences, fit_sentences, split_sentences
from .presets import ModelPreset, family_defaults, find_preset, generic_generation

logger = logging.getLogger(__name__)

META_FILE = "summarizer_meta.json"
TRUNCATION_MODES = ("tokens", "sentences")
LONG_DOC_MODES = ("truncate", "chunk")


class ModelLoadError(OSError):
    """Raised when a model cannot be loaded (missing, or no internet to download it)."""


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AbstractiveSummarizer:
    """Wraps a sequence-to-sequence model for news summarisation."""

    kind = "abstractive"

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        max_input_tokens: int | None = None,
        truncation: str = "tokens",
        long_doc: str = "truncate",
        num_beams: int | None = None,
        max_summary_tokens: int | None = None,
        min_summary_tokens: int | None = None,
        generation: dict | None = None,
        prefix: str | None = None,
        dtype: str = "float32",
    ) -> None:
        if truncation not in TRUNCATION_MODES:
            raise ValueError(f"truncation must be one of {TRUNCATION_MODES}")
        if long_doc not in LONG_DOC_MODES:
            raise ValueError(f"long_doc must be one of {LONG_DOC_MODES}")
        self.truncation, self.long_doc = truncation, long_doc
        self.preset: ModelPreset | None = find_preset(model_name_or_path)
        source = self.preset.hub_id if self.preset else model_name_or_path
        self.source = source
        self.name = self.preset.key if self.preset else Path(str(model_name_or_path).rstrip("/\\")).name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(source)
        except Exception as err:  # network / missing files / bad id
            raise ModelLoadError(
                f"Could not load model '{source}': {err}\n"
                "Models are downloaded from the Hugging Face Hub on first use, so an internet connection is "
                "required. For a local model pass its directory."
            ) from err
        self.device = pick_device(device)
        self.model.to(self.device).eval()
        if dtype != "float32":
            self.model.to(getattr(torch, dtype))

        meta = self._read_meta(source)
        family, default_prefix, default_gen = family_defaults(self.model.config.model_type)
        task_params = (getattr(self.model.config, "task_specific_params", None) or {}).get("summarization", {})
        if self.preset:
            family, default_prefix, default_gen = self.preset.family, self.preset.prefix, dict(self.preset.generation)
        elif task_params:
            default_gen.update({k: v for k, v in task_params.items() if k != "prefix"})
            default_prefix = task_params.get("prefix", default_prefix)
        self.family = family
        self.prefix = prefix if prefix is not None else meta.get("prefix", default_prefix)
        gen = {**default_gen, **meta.get("generation", {})}
        gen.update(generation or {})
        if num_beams is not None:
            gen["num_beams"] = num_beams
        if max_summary_tokens is not None:
            gen["max_length"] = max_summary_tokens
        if min_summary_tokens is not None:
            gen["min_length"] = min_summary_tokens
        self.generation = self._finalize_generation(gen)
        self.max_input_tokens = int(max_input_tokens or meta.get("max_input_tokens") or self._default_max_input())

        self._n_special = self.tokenizer.num_special_tokens_to_add(pair=False)
        self._n_prefix = len(self.tokenizer(self.prefix, add_special_tokens=False)["input_ids"]) if self.prefix else 0

    # ----------------------------------------------------------------------------- configuration
    @staticmethod
    def _read_meta(source: str) -> dict:
        path = Path(source) / META_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    @staticmethod
    def _finalize_generation(gen: dict) -> dict:
        gen = {k: v for k, v in (gen or generic_generation()).items() if v is not None}
        if gen.get("num_beams", 1) <= 1:                     # beam-only options are meaningless for greedy search
            gen.pop("early_stopping", None)
            gen.pop("length_penalty", None)
        if "max_length" in gen and "min_length" in gen and gen["min_length"] >= gen["max_length"]:
            gen["min_length"] = max(0, gen["max_length"] - 1)
        return gen

    def _default_max_input(self) -> int:
        """Encoder limit for models without a preset: smallest sane value among the model's own limits."""
        candidates = []
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if tok_max and tok_max < 100_000:
            candidates.append(int(tok_max))
        for attr in ("max_position_embeddings", "n_positions"):
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        return min(candidates) if candidates else 512

    @property
    def num_parameters_millions(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def info(self) -> dict:
        return {"model": self.source, "family": self.family,
                "parameters_millions": round(self.num_parameters_millions, 1),
                "device": str(self.device), "max_input_tokens": self.max_input_tokens, "prefix": self.prefix,
                "truncation": self.truncation, "long_doc": self.long_doc, "generation": self.generation}

    # ----------------------------------------------------------------------------- tokens
    def _encode_lengths(self, texts: list[str], with_prefix: bool = True) -> list[int]:
        if not texts:
            return []
        batch = [(self.prefix if with_prefix else "") + t for t in texts]
        ids = self.tokenizer(batch, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        return [len(x) for x in ids]

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Number of encoder tokens each text needs (prefix and special tokens included)."""
        return self._encode_lengths(texts)

    def truncation_rate(self, texts: list[str]) -> float:
        """Share of texts longer than the model's input limit (they lose content when truncated)."""
        lengths = self.token_lengths(texts)
        return sum(n > self.max_input_tokens for n in lengths) / max(len(lengths), 1)

    def _budget(self) -> int:
        """Tokens available for article text once prefix and special tokens are accounted for."""
        return max(8, self.max_input_tokens - self._n_special - self._n_prefix)

    def _prepare(self, text: str) -> str:
        """Apply sentence-aware truncation (when selected) and the task prefix."""
        if self.truncation == "sentences":
            sentences = split_sentences(text)
            lengths = self._encode_lengths(sentences, with_prefix=False)
            lengths = [n - self._n_special for n in lengths]
            text = fit_sentences(sentences, lengths, self._budget())
        return self.prefix + text

    # ----------------------------------------------------------------------------- generation
    @torch.inference_mode()
    def _generate(self, prepared: list[str]) -> list[str]:
        enc = self.tokenizer(prepared, truncation=True, max_length=self.max_input_tokens, padding=True,
                             return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, **self.generation)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [d.replace("<n>", "\n").strip() for d in decoded]

    def _generate_batches(self, prepared: list[str], batch_size: int, progress: bool = False) -> list[str]:
        """Length-sorted batching; output order matches input order."""
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]), reverse=True)
        results: list[str] = [""] * len(prepared)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            for i, summary in zip(idx, self._generate([prepared[i] for i in idx])):
                results[i] = summary
            if progress:
                logger.info("  %s: %d/%d", self.name, min(start + batch_size, len(order)), len(order))
        return results

    def _summarize_long(self, text: str, batch_size: int, depth: int = 0) -> str:
        """Chunk -> summarise every chunk -> summarise the concatenated chunk summaries."""
        sentences = split_sentences(text)
        lengths = [n - self._n_special for n in self._encode_lengths(sentences, with_prefix=False)]
        chunks = chunk_sentences(sentences, lengths, self._budget())
        partial = self._generate_batches([self.prefix + c for c in chunks], batch_size)
        if len(chunks) == 1:
            return partial[0]
        merged = " ".join(p for p in partial if p)
        if depth < 2 and self.token_lengths([merged])[0] > self.max_input_tokens:
            return self._summarize_long(merged, batch_size, depth + 1)
        return self._generate([self._prepare(merged)])[0]

    def summarize(self, texts: list[str], batch_size: int = 8, progress: bool = False) -> list[str]:
        """Summarise articles (already cleaned). Returns one summary per input, in order."""
        results: list[str] = [""] * len(texts)
        todo = [i for i, t in enumerate(texts) if t and t.strip()]
        lengths = dict(zip(todo, self.token_lengths([texts[i] for i in todo])))
        long_idx = [i for i in todo if self.long_doc == "chunk" and lengths[i] > self.max_input_tokens]
        short_idx = [i for i in todo if i not in set(long_idx)]

        prepared = [self._prepare(texts[i]) for i in short_idx]
        for i, summary in zip(short_idx, self._generate_batches(prepared, batch_size, progress)):
            results[i] = summary
        for i in long_idx:
            results[i] = self._summarize_long(texts[i], batch_size)
        return results
