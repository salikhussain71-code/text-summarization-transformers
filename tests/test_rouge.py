import numpy as np
import pytest

from text_summarization.rouge import (
    aggregate,
    bootstrap_ci,
    compute_rouge,
    novel_ngram_ratio,
    score_pairs,
    summary_stats,
)


def test_identical_text_scores_100():
    s = compute_rouge(["The cat sat on the mat."], ["The cat sat on the mat."], n_boot=20)
    assert all(s[t] == pytest.approx(100.0) for t in ("rouge1", "rouge2", "rougeL", "rougeLsum"))


def test_disjoint_text_scores_0_and_empty_prediction_is_safe():
    assert compute_rouge(["alpha beta"], ["gamma delta"], n_boot=10)["rouge1"] == 0.0
    assert compute_rouge([""], ["something here"], n_boot=10)["rougeL"] == 0.0


def test_stemming_matches_word_variants():
    with_stem = score_pairs(["the cats are running"], ["the cat runs"], use_stemmer=True)["rouge1_f"][0]
    without = score_pairs(["the cats are running"], ["the cat runs"], use_stemmer=False)["rouge1_f"][0]
    assert with_stem > without


def test_known_value():
    # 5 of 6 unigrams overlap in both directions -> F1 = 5/6
    s = score_pairs(["the cat sat on the mat"], ["the cat sat on a mat"])
    assert s["rouge1_f"][0] == pytest.approx(100 * 5 / 6)


def test_multi_sentence_summary_level_rouge():
    pred = "A man was arrested. Police said he was calm."
    ref = "Police said he was calm.\nA man was arrested."
    s = score_pairs([pred], [ref])
    assert s["rougeLsum_f"][0] == pytest.approx(100.0)       # sentence order does not matter for Lsum


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_pairs(["a"], ["a", "b"])


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = np.random.default_rng(0).normal(40, 5, size=60)
    lo, hi = bootstrap_ci(values, 500, seed=1)
    assert lo < values.mean() < hi and (lo, hi) == bootstrap_ci(values, 500, seed=1)
    assert bootstrap_ci(np.array([3.0]), 10) == (3.0, 3.0) and bootstrap_ci(np.array([]), 10) == (0.0, 0.0)


def test_aggregate_has_all_keys():
    agg = aggregate(score_pairs(["a b c", "d e f"], ["a b c", "x y z"]), n_boot=50)
    assert agg["rouge1"] == pytest.approx(50.0)
    assert {"rouge2_ci_low", "rougeL_ci_high", "rougeLsum_precision", "rouge1_recall"} <= set(agg)


def test_novel_ngram_ratio():
    article = "the mayor announced a new transport plan"
    assert novel_ngram_ratio("the mayor announced a new plan", article, 1) == 0.0
    assert novel_ngram_ratio("completely different words here", article, 2) == 1.0
    assert novel_ngram_ratio("", article, 2) == 0.0
    assert 0 < novel_ngram_ratio("the mayor announced something", article, 2) < 1


def test_summary_stats():
    stats = summary_stats(["one two three"], ["one two three four five six"], ["one two"])
    assert stats["summary_words"] == 3 and stats["reference_words"] == 2
    assert stats["compression"] == pytest.approx(0.5) and stats["novel_bigrams"] == 0.0
