from text_summarization.preprocessing import (
    chunk_sentences,
    clean_article,
    clean_summary,
    fit_sentences,
    split_sentences,
)


def test_cnn_prefix_removed():
    assert clean_article("LONDON, England (CNN) -- The bishop spoke.") == "The bishop spoke."
    assert clean_article("(CNN) -- Hello   there.") == "Hello there."


def test_cnn_marker_late_in_text_is_kept():
    text = "A" * 120 + " said it on (CNN) -- later."
    assert "(CNN)" in clean_article(text)


def test_daily_mail_header_removed():
    raw = ("By . Daily Mail Reporter . PUBLISHED: . 05:12 EST, 12 March 2013 . | . UPDATED: . "
           "08:33 EST, 12 March 2013 . A man was arrested . Police said he was calm .")
    assert clean_article(raw) == "A man was arrested . Police said he was calm ."


def test_last_updated_line_removed():
    assert clean_article("Last updated at 9:08 AM on 19th February 2012 . The cat sat.") == "The cat sat."


def test_clean_article_handles_missing_values():
    assert clean_article(None) == "" and clean_article("  \n ") == ""


def test_clean_summary_one_highlight_per_line():
    assert clean_summary("First point\n\n  Second   point \r\nThird") == "First point\nSecond point\nThird"


def test_split_sentences_protects_abbreviations_and_numbers():
    text = 'Dr. Smith met Mr. Jones in the U.S. on Jan. 5. The price rose 3.5 percent. "Wow!" he said.'
    assert split_sentences(text) == ["Dr. Smith met Mr. Jones in the U.S. on Jan. 5.",
                                     "The price rose 3.5 percent.", '"Wow!" he said.']


def test_split_sentences_newlines_and_empty():
    assert split_sentences("First line\nSecond line") == ["First line", "Second line"]
    assert split_sentences("") == [] and split_sentences("   ") == []
    assert split_sentences("John F. Kennedy spoke.") == ["John F. Kennedy spoke."]


def test_fit_sentences_respects_budget_but_keeps_first():
    sents, lens = ["a b", "c d e", "f"], [2, 3, 1]
    assert fit_sentences(sents, lens, 5) == "a b c d e"
    assert fit_sentences(sents, lens, 4) == "a b"
    assert fit_sentences(["very long first sentence"], [50], 5) == "very long first sentence"
    assert fit_sentences([], [], 5) == ""


def test_chunk_sentences_cover_everything_within_budget():
    sents = [f"s{i}" for i in range(10)]
    lens = [3, 4, 2, 5, 1, 1, 6, 2, 2, 3]
    chunks = chunk_sentences(sents, lens, 7)
    assert " ".join(chunks).split() == sents                      # nothing lost, order preserved
    assert all(sum(lens[sents.index(s)] for s in c.split()) <= 7 for c in chunks)
    assert chunk_sentences(["x"], [100], 5) == ["x"]               # oversized sentence -> own chunk
    assert chunk_sentences([], [], 5) == []
