import pytest

from text_summarization.testing import build_tiny_seq2seq, main


def test_both_families_are_created(tmp_path):
    for family in ("t5", "bart"):
        path = build_tiny_seq2seq(tmp_path / family, family)
        assert (path / "model.safetensors").exists() and (path / "tokenizer.json").exists()


def test_invalid_family(tmp_path):
    with pytest.raises(ValueError):
        build_tiny_seq2seq(tmp_path / "x", "gpt")


def test_cli(tmp_path, capsys):
    assert main(["--family", "t5", "--out", str(tmp_path / "m")]) == 0
    assert "random weights" in capsys.readouterr().out
