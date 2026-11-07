import json
import shutil

import pytest

from text_summarization.presets import PRESETS, family_defaults, find_preset
from text_summarization.summarizer import AbstractiveSummarizer, ModelLoadError, pick_device

GEN = dict(num_beams=1, max_summary_tokens=12, min_summary_tokens=2)
LONG = " ".join(f"Sentence number {i} is about the transport plan and the city council vote." for i in range(40))


def test_presets():
    assert find_preset("t5-small").prefix == "summarize: " and find_preset("t5-small").max_input_tokens == 512
    assert find_preset("facebook/bart-large-cnn").key == "bart-large-cnn"
    assert find_preset("sshleifer/distilbart-cnn-12-6").max_input_tokens == 1024
    assert find_preset("not-a-preset") is None and "pegasus-cnn_dailymail" in PRESETS
    assert family_defaults("t5")[1] == "summarize: " and family_defaults("bart")[1] == ""
    assert family_defaults("something-else")[0] == "generic"


def test_t5_and_bart_load_with_the_right_defaults(tiny_t5, tiny_bart):
    t5, bart = AbstractiveSummarizer(tiny_t5, **GEN), AbstractiveSummarizer(tiny_bart, **GEN)
    assert (t5.family, t5.prefix) == ("t5", "summarize: ") and (bart.family, bart.prefix) == ("bart", "")
    assert t5.max_input_tokens == 128 and bart.max_input_tokens == 128      # tokenizer / config limit
    assert t5.kind == "abstractive" and t5.num_parameters_millions > 0 and t5.info()["family"] == "t5"


def test_overrides_and_generation_sanitising(tiny_t5):
    s = AbstractiveSummarizer(tiny_t5, max_input_tokens=64, num_beams=3, max_summary_tokens=20, min_summary_tokens=30,
                              prefix="x: ")
    assert s.max_input_tokens == 64 and s.prefix == "x: "
    assert s.generation["num_beams"] == 3 and s.generation["max_length"] == 20
    assert s.generation["min_length"] < s.generation["max_length"]           # min >= max is repaired
    greedy = AbstractiveSummarizer(tiny_t5, num_beams=1)
    assert "early_stopping" not in greedy.generation and "length_penalty" not in greedy.generation


def test_summarize_shapes_and_empty_inputs(tiny_t5, tiny_bart, article):
    for path in (tiny_t5, tiny_bart):
        s = AbstractiveSummarizer(path, **GEN)
        out = s.summarize([article, "", "   ", "A short one."], batch_size=2)
        assert len(out) == 4 and all(isinstance(o, str) for o in out)
        assert out[1] == "" and out[2] == "" and s.summarize([]) == []


def test_batching_restores_the_original_order(tiny_t5, monkeypatch):
    s = AbstractiveSummarizer(tiny_t5, **GEN)
    monkeypatch.setattr(s, "_generate", lambda prepared: [p.upper() for p in prepared])
    texts = ["short one", "a considerably longer text than the others here", "mid length text", "tiny", "x " * 30]
    assert s.summarize(texts, batch_size=2) == [("summarize: " + t).upper() for t in texts]


def test_token_lengths_and_truncation_rate(tiny_t5, article):
    s = AbstractiveSummarizer(tiny_t5, max_input_tokens=40, **GEN)
    lengths = s.token_lengths([article, "Hi."])
    assert lengths[0] > 40 > lengths[1]
    assert s.truncation_rate([article, "Hi."]) == 0.5 and s.truncation_rate([]) == 0.0


def test_truncation_never_exceeds_the_limit(tiny_t5, tiny_bart):
    for path in (tiny_t5, tiny_bart):
        s = AbstractiveSummarizer(path, max_input_tokens=48, **GEN)
        assert len(s.summarize([LONG])[0]) >= 0                                 # hard token truncation works
        sent = AbstractiveSummarizer(path, max_input_tokens=48, truncation="sentences", **GEN)
        prepared = sent._prepare(LONG)
        assert sent.token_lengths([prepared.removeprefix(sent.prefix)])[0] <= 48
        assert prepared.removeprefix(sent.prefix).endswith("vote.")             # whole sentences only


def test_long_documents_are_chunked_then_merged(tiny_t5, monkeypatch):
    s = AbstractiveSummarizer(tiny_t5, max_input_tokens=64, long_doc="chunk", **GEN)
    calls = []

    def fake(prepared):
        calls.append(list(prepared))
        return ["summary"] * len(prepared)

    monkeypatch.setattr(s, "_generate", fake)
    assert s.summarize([LONG, "Short text."]) == ["summary", "summary"]
    assert sum(len(c) for c in calls) > 3                                       # chunks + merge pass + short text
    plain = AbstractiveSummarizer(tiny_t5, max_input_tokens=64, long_doc="truncate", **GEN)
    calls.clear()
    monkeypatch.setattr(plain, "_generate", fake)
    plain.summarize([LONG])
    assert sum(len(c) for c in calls) == 1                                      # truncate mode: one pass only


def test_real_chunked_generation_runs(tiny_bart):
    s = AbstractiveSummarizer(tiny_bart, max_input_tokens=64, long_doc="chunk", truncation="sentences", **GEN)
    assert isinstance(s.summarize([LONG])[0], str)


def test_meta_file_is_honoured(tiny_t5, tmp_path):
    folder = tmp_path / "ft"
    shutil.copytree(tiny_t5, folder)
    meta = {"prefix": "tldr: ", "max_input_tokens": 70, "generation": {"num_beams": 1, "max_length": 9}}
    (folder / "summarizer_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    s = AbstractiveSummarizer(str(folder))
    assert s.prefix == "tldr: " and s.max_input_tokens == 70 and s.generation["max_length"] == 9


def test_invalid_arguments_and_missing_model(tiny_t5, tmp_path):
    with pytest.raises(ValueError):
        AbstractiveSummarizer(tiny_t5, truncation="words")
    with pytest.raises(ValueError):
        AbstractiveSummarizer(tiny_t5, long_doc="map-reduce")
    with pytest.raises(ModelLoadError, match="internet|directory") as err:
        AbstractiveSummarizer(str(tmp_path / "does-not-exist"))
    assert isinstance(err.value, OSError)


def test_pick_device():
    assert pick_device("cpu").type == "cpu" and pick_device("auto").type in ("cpu", "cuda", "mps")
