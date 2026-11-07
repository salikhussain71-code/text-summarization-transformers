"""Extractive summarisation baselines (bonus): Lead-N and TextRank.

* **Lead-N** - the first N sentences. A famously strong baseline for news.
* **TextRank** - graph-based sentence ranking (Mihalcea & Tarau, 2004). Sentences are nodes, edges are
  TF-IDF cosine similarities, PageRank scores the nodes, and the top N sentences are returned in
  their original order. Implemented with NumPy only (no extra dependencies).
"""

from __future__ import annotations

import re

import numpy as np

from .preprocessing import split_sentences

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being below between
both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his
how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves said says also will just one two new""".split())

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _content_words(sentence: str) -> list[str]:
    return [w for w in _WORD.findall(sentence.lower()) if w not in STOPWORDS and len(w) > 1]


def lead_n(text: str, n: int = 3) -> str:
    """First ``n`` sentences of ``text``."""
    return " ".join(split_sentences(text)[:n])


def similarity_matrix(sentences: list[str]) -> np.ndarray:
    """Cosine similarity of TF-IDF sentence vectors (zero diagonal)."""
    tokens = [_content_words(s) for s in sentences]
    vocab = {w: i for i, w in enumerate(sorted({w for t in tokens for w in t}))}
    n = len(sentences)
    if not vocab:
        return np.zeros((n, n))
    tf = np.zeros((n, len(vocab)))
    for i, words in enumerate(tokens):
        for w in words:
            tf[i, vocab[w]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log((n + 1) / (df + 1)) + 1.0
    vectors = tf * idf
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, 0.0)
    return sim


def pagerank(weights: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """PageRank by power iteration on a non-negative weight matrix."""
    n = weights.shape[0]
    if n == 0:
        return np.array([])
    row_sums = weights.sum(axis=1, keepdims=True)
    transition = np.where(row_sums > 0, weights / np.where(row_sums == 0, 1, row_sums), 1.0 / n)  # dangling -> uniform
    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = (1 - damping) / n + damping * (transition.T @ scores)
        if np.abs(new - scores).sum() < tol:
            scores = new
            break
        scores = new
    return scores


def textrank_scores(sentences: list[str], damping: float = 0.85) -> np.ndarray:
    """TextRank score of every sentence."""
    return pagerank(similarity_matrix(sentences), damping=damping)


def textrank_summary(text: str, n: int = 3, lead_bias: float = 0.0, damping: float = 0.85) -> str:
    """Top-``n`` TextRank sentences in original order.

    ``lead_bias`` > 0 down-weights later sentences (score / (1 + lead_bias * position)), a useful
    prior for news where the first sentences carry the key facts.
    """
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return " ".join(sentences)
    scores = textrank_scores(sentences, damping)
    if lead_bias > 0:
        scores = scores / (1.0 + lead_bias * np.arange(len(sentences)))
    chosen = sorted(np.argsort(-scores, kind="stable")[:n])
    return " ".join(sentences[i] for i in chosen)


class _Extractive:
    kind = "extractive"

    def summarize(self, texts: list[str], **_: object) -> list[str]:
        return [self.summarize_one(t) for t in texts]

    def summarize_one(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LeadSummarizer(_Extractive):
    """Lead-N baseline."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.name = f"Lead-{n}"

    def summarize_one(self, text: str) -> str:
        return lead_n(text, self.n)


class TextRankSummarizer(_Extractive):
    """TextRank extractive summariser."""

    def __init__(self, n: int = 3, lead_bias: float = 0.0) -> None:
        self.n, self.lead_bias = n, lead_bias
        self.name = f"TextRank-{n}" + (" (+lead bias)" if lead_bias else "")

    def summarize_one(self, text: str) -> str:
        return textrank_summary(text, self.n, self.lead_bias)
