import numpy as np

from text_summarization.extractive import (
    LeadSummarizer,
    TextRankSummarizer,
    lead_n,
    pagerank,
    similarity_matrix,
    textrank_summary,
)


def test_lead_n(article):
    assert lead_n(article, 2) == ("The mayor announced a new transport plan on Monday. "
                                  "The plan will add forty buses to the city network.")
    assert lead_n("", 3) == ""


def test_textrank_returns_original_order_and_n_sentences(article):
    summary = textrank_summary(article, 3)
    sentences = [s for s in article.split(". ") if s]
    positions = [article.find(s.rstrip(".")) for s in sentences if s.rstrip(".") in summary]
    assert positions == sorted(positions)
    assert summary.count(".") == 3


def test_textrank_prefers_central_sentences(article):
    summary = textrank_summary(article, 2)
    assert "transport plan" in summary and "weather" not in summary


def test_textrank_short_inputs():
    assert textrank_summary("Only one sentence.", 3) == "Only one sentence."
    assert textrank_summary("", 3) == ""
    assert textrank_summary("12 34 . 56 78 . 90 12 . 34 56 .", 2)  # no content words -> must not crash


def test_lead_bias_moves_selection_to_the_start():
    text = " ".join(f"Sentence number {i} talks about topic{i % 2} and some detail{i}." for i in range(10))
    plain = textrank_summary(text, 2)
    biased = textrank_summary(text, 2, lead_bias=5.0)
    assert biased.split(".")[0].split()[2] in ("0", "1", "2", "3")
    assert isinstance(plain, str)


def test_similarity_matrix_is_symmetric_with_zero_diagonal(article):
    from text_summarization.preprocessing import split_sentences

    sim = similarity_matrix(split_sentences(article))
    assert np.allclose(sim, sim.T) and np.allclose(np.diag(sim), 0) and (sim >= 0).all()


def test_pagerank_is_a_distribution_and_handles_dangling_nodes():
    w = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.0]])
    scores = pagerank(w)
    assert scores.shape == (3,) and np.isclose(scores.sum(), 1.0, atol=1e-6)
    assert pagerank(np.zeros((0, 0))).size == 0


def test_summarizer_classes(article):
    lead, rank = LeadSummarizer(2), TextRankSummarizer(2, lead_bias=0.3)
    assert lead.name == "Lead-2" and "lead bias" in rank.name and lead.kind == "extractive"
    out = lead.summarize([article, article], batch_size=8)
    assert out[0] == out[1] and len(rank.summarize([article])) == 1
    assert textrank_summary(article, 3) == textrank_summary(article, 3)   # deterministic
