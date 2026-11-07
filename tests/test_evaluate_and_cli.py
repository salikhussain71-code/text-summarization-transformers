import json

import pandas as pd

from text_summarization import evaluate, summarize

README = f"# t\n{evaluate.RESULTS_START}\nPLACEHOLDER_TEXT\n{evaluate.RESULTS_END}\n"
FAST = ["--max-summary-tokens", "16", "--min-summary-tokens", "3", "--num-beams", "1", "--no-plots"]


def test_end_to_end_evaluation(isolated, sample_dir, tiny_t5, tiny_bart):
    (isolated / "README.md").write_text(README, encoding="utf-8")
    code = evaluate.main(["--data-dir", str(sample_dir), "--n", "10", "--models", tiny_t5, tiny_bart,
                          "--max-summary-tokens", "16", "--min-summary-tokens", "3", "--num-beams", "1",
                          "--bootstrap", "50", "--examples", "2"])
    assert code == 0
    table = pd.read_csv(isolated / "reports/metrics/rouge_results.csv")
    assert len(table) == 4 and set(table["kind"]) == {"extractive", "abstractive"}
    assert table["rougeL"].between(0, 100).all() and (table["rougeL_ci_low"] <= table["rougeL_ci_high"]).all()
    assert table.loc[table["system"] == "Lead-3", "rougeL"].iloc[0] > 30        # lead is strong on this sample

    preds = sorted(p.name for p in (isolated / "reports/predictions").glob("*.csv"))
    assert len(preds) == 4 and "Lead-3.csv" in preds
    one = pd.read_csv(isolated / "reports/predictions/Lead-3.csv")
    assert {"id", "reference", "prediction", "rougeL_f"} <= set(one.columns) and len(one) == 10

    for f in ("rouge_comparison.png", "speed_vs_quality.png", "summary_lengths.png", "abstractiveness.png",
              "dataset_overview.png"):
        assert (isolated / "reports/figures" / f).exists()
    assert "Example 2" in (isolated / "reports/examples.md").read_text(encoding="utf-8")
    run = json.loads((isolated / "reports/metrics/run_summary.json").read_text())
    assert run["n_articles"] == 10 and run["environment"]["torch"] and len(run["models"]) == 2
    readme = (isolated / "README.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER_TEXT" not in readme and "ROUGE-L 95% CI" in readme and "limit 128 tokens" in readme


def test_baselines_only_and_split_selection(isolated, sample_dir):
    code = evaluate.main(["--data-dir", str(sample_dir), "--split", "validation", "--n", "0", "--models",
                          "--baselines", "lead", "textrank", "textrank-lead", "--sentences", "2", "--no-plots",
                          "--no-update-readme"])
    assert code == 0
    table = pd.read_csv(isolated / "reports/metrics/rouge_results.csv")
    assert len(table) == 3 and set(table["kind"]) == {"extractive"}
    assert json.loads((isolated / "reports/metrics/run_summary.json").read_text())["n_articles"] == 30


def test_unavailable_model_is_skipped_not_fatal(isolated, sample_dir, tiny_t5, tmp_path):
    code = evaluate.main(["--data-dir", str(sample_dir), "--n", "5", "--models", str(tmp_path / "missing"), tiny_t5,
                          "--baselines", *FAST, "--no-update-readme"])
    assert code == 0
    assert len(pd.read_csv(isolated / "reports/metrics/rouge_results.csv")) == 1


def test_sentence_truncation_and_chunking_flags(isolated, sample_dir, tiny_bart):
    code = evaluate.main(["--data-dir", str(sample_dir), "--n", "6", "--models", tiny_bart, "--baselines",
                          "--truncation", "sentences", "--long-doc", "chunk", "--max-input-tokens", "48", *FAST,
                          "--no-update-readme"])
    assert code == 0
    assert json.loads((isolated / "reports/metrics/run_summary.json").read_text())["long_doc"] == "chunk"


def test_errors(isolated, tmp_path):
    assert evaluate.main(["--data-dir", str(tmp_path)]) == 1                                  # no CSV
    assert evaluate.main(["--data", str(tmp_path / "x.csv")]) == 1                            # missing file
    assert evaluate.main(["--data-dir", "data/sample", "--models", "--baselines"]) == 1       # nothing to run


# ----------------------------------------------------------------------------------- summarize CLI
def test_cli_extractive_methods(capsys, article):
    assert summarize.main(["--method", "lead", "--sentences", "1", "(CNN) -- " + article]) == 0
    out = capsys.readouterr().out
    assert "The mayor announced a new transport plan on Monday." in out and "(CNN)" not in out
    assert summarize.main(["--method", "textrank", "--sentences", "2", article]) == 0
    assert "TextRank-2" in capsys.readouterr().out


def test_cli_abstractive_csv_in_and_out(tmp_path, tiny_t5, capsys, article):
    src, dst = tmp_path / "in.csv", tmp_path / "out" / "sum.csv"
    pd.DataFrame({"content": [article, None, "A second article with enough words in it."]}).to_csv(src, index=False)
    code = summarize.main(["--model", tiny_t5, "--csv", str(src), "--text-col", "content", "--out", str(dst),
                           "--max-summary-tokens", "12", "--min-summary-tokens", "2", "--num-beams", "1"])
    assert code == 0
    saved = pd.read_csv(dst)
    assert len(saved) == 2 and {"source", "words_in", "words_out", "summary"} <= set(saved.columns)
    capsys.readouterr()


def test_cli_files_and_errors(tmp_path, capsys, article):
    f = tmp_path / "a.txt"
    f.write_text(article, encoding="utf-8")
    assert summarize.main(["--method", "lead", "--file", str(f)]) == 0 and "a.txt" in capsys.readouterr().out
    pd.DataFrame({"x": ["y"]}).to_csv(tmp_path / "c.csv", index=False)
    assert summarize.main(["--method", "lead", "--csv", str(tmp_path / "c.csv"), "--text-col", "nope"]) == 1
    assert summarize.main([]) == 1                                                     # nothing to summarise
    assert summarize.main(["--model", str(tmp_path / "missing"), "text here"]) == 1    # model cannot be loaded
    capsys.readouterr()
