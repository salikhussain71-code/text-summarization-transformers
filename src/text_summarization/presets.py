"""Known summarisation checkpoints and their recommended settings.

``max_input_tokens`` is the encoder limit; longer articles are truncated (or chunked). Generation
settings follow the configuration published with each checkpoint. Any other Hugging Face model id or
local directory also works - sensible defaults are then derived from the model's own config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_BART_GEN = dict(num_beams=4, max_length=142, min_length=56, length_penalty=2.0, no_repeat_ngram_size=3,
                 early_stopping=True)
_T5_GEN = dict(num_beams=4, max_length=200, min_length=30, length_penalty=2.0, no_repeat_ngram_size=3,
               early_stopping=True)
_PEGASUS_GEN = dict(num_beams=8, max_length=128, min_length=32, length_penalty=0.8, no_repeat_ngram_size=3)
_GENERIC_GEN = dict(num_beams=4, max_length=128, min_length=30, length_penalty=1.0, no_repeat_ngram_size=3,
                    early_stopping=True)


@dataclass(frozen=True)
class ModelPreset:
    key: str
    hub_id: str
    family: str                      # "bart" | "t5" | "pegasus" | "generic"
    max_input_tokens: int
    prefix: str = ""
    generation: dict = field(default_factory=dict)
    note: str = ""


PRESETS: dict[str, ModelPreset] = {
    p.key: p
    for p in (
        ModelPreset("distilbart-cnn-12-6", "sshleifer/distilbart-cnn-12-6", "bart", 1024, "", _BART_GEN,
                    "Distilled BART fine-tuned on CNN/DM: fast, CPU friendly (~1.2 GB)"),
        ModelPreset("bart-large-cnn", "facebook/bart-large-cnn", "bart", 1024, "", _BART_GEN,
                    "BART-large fine-tuned on CNN/DM: strong, slower (~1.6 GB)"),
        ModelPreset("t5-small", "t5-small", "t5", 512, "summarize: ", _T5_GEN,
                    "T5-small, multi-task pre-trained incl. CNN/DM summarisation (~240 MB)"),
        ModelPreset("t5-base", "t5-base", "t5", 512, "summarize: ", _T5_GEN,
                    "T5-base, multi-task pre-trained incl. CNN/DM summarisation (~900 MB)"),
        ModelPreset("pegasus-cnn_dailymail", "google/pegasus-cnn_dailymail", "pegasus", 1024, "", _PEGASUS_GEN,
                    "PEGASUS fine-tuned on CNN/DM (~2.2 GB)"),
    )
}
_BY_HUB_ID = {p.hub_id: p for p in PRESETS.values()}


def find_preset(name: str) -> ModelPreset | None:
    """Preset for a short key (``t5-small``) or a hub id (``facebook/bart-large-cnn``), else ``None``."""
    return PRESETS.get(name) or _BY_HUB_ID.get(name)


def generic_generation() -> dict:
    return dict(_GENERIC_GEN)


def family_defaults(model_type: str) -> tuple[str, str, dict]:
    """``(family, prefix, generation)`` guessed from a model config's ``model_type``."""
    if model_type in ("t5", "mt5", "umt5", "longt5"):
        return "t5", "summarize: ", dict(_T5_GEN)
    if model_type == "pegasus":
        return "pegasus", "", dict(_PEGASUS_GEN)
    if model_type in ("bart", "mbart", "led", "blenderbot"):
        return "bart", "", dict(_BART_GEN)
    return "generic", "", dict(_GENERIC_GEN)
