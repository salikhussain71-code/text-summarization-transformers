"""Text preprocessing for summarisation.

* ``clean_article`` removes CNN / Daily Mail boilerplate (``LONDON, England (CNN) --``,
  ``By . Reporter . PUBLISHED: ... UPDATED: ...``) and normalises whitespace.
* ``split_sentences`` is a dependency-free sentence splitter that protects common abbreviations.
* ``fit_sentences`` / ``chunk_sentences`` implement the **input-length handling** asked for by the task:
  keep a document within a model's token budget, either by cutting at a sentence boundary
  (truncation) or by splitting it into several budget-sized chunks (long-document mode).

The length helpers take pre-computed token counts, so they are independent of any tokenizer.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")
# "By . Daily Mail Reporter . PUBLISHED: . 05:12 EST, 12 March 2013 . | . UPDATED: . 08:33 EST, 12 March 2013 ."
_DM_HEADER = re.compile(
    r"^\s*(?:By\s+\.\s+.{1,100}?\s+\.\s+)?PUBLISHED:\s+\.?\s*.{1,60}?\d{4}\s+\.\s+\|\s+\.\s+"
    r"UPDATED:\s+\.?\s*.{1,60}?\d{4}\s+\.\s+",
    flags=re.DOTALL,
)
_DM_PUBLISHED_ONLY = re.compile(r"^\s*(?:By\s+\.\s+.{1,100}?\s+\.\s+)?PUBLISHED:\s+\.?\s*.{1,60}?\d{4}\s+\.\s+",
                                flags=re.DOTALL)
_LAST_UPDATED = re.compile(r"^\s*Last updated at\s+.{1,40}?\d{4}\s*\.?\s+", flags=re.IGNORECASE | re.DOTALL)
# "LONDON, England (CNN) -- ..." / "(CNN) -- ..." (only within the first 80 characters)
_CNN_PREFIX = re.compile(r"^.{0,80}?\(CNN\)\s*(?:--|-|\u2014|:)?\s*", flags=re.DOTALL)
_BY_LINE = re.compile(r"^\s*By\s+\.\s+[A-Z][^.]{1,60}\s+\.\s+(?=[A-Z])")


def clean_article(text: object) -> str:
    """Strip news-agency boilerplate and normalise whitespace."""
    text = "" if text is None else str(text)
    text = _DM_HEADER.sub("", text, count=1)
    text = _DM_PUBLISHED_ONLY.sub("", text, count=1)
    text = _LAST_UPDATED.sub("", text, count=1)
    text = _CNN_PREFIX.sub("", text, count=1)
    text = _BY_LINE.sub("", text, count=1)
    return _WHITESPACE.sub(" ", text).strip()


def clean_summary(text: object) -> str:
    """Normalise a reference summary: one sentence/highlight per line, no empty lines."""
    text = "" if text is None else str(text)
    lines = [_WHITESPACE.sub(" ", ln).strip() for ln in re.split(r"[\r\n]+", text)]
    return "\n".join(ln for ln in lines if ln)


# --------------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------------
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Mt", "Gen", "Col", "Lt", "Sgt", "Capt", "Cmdr", "Gov",
    "Sen", "Rep", "Rev", "Hon", "Inc", "Corp", "Ltd", "Co", "vs", "etc", "No", "Jan", "Feb", "Mar", "Apr", "Jun",
    "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "Fig", "Vol", "approx", "dept", "est",
)
_ABBR = re.compile(r"\b(" + "|".join(_ABBREVIATIONS) + r")\.", flags=re.IGNORECASE)
_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}")                 # U.S.  a.m.  e.g.
_INITIAL = re.compile(r"\b([A-Z])\.(?=\s+[A-Z])")             # "John F. Kennedy"
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")                     # 3.5
_BOUNDARY = re.compile(r"""([.!?]["'\u201d\u2019)\]]*)\s+(?=["'\u201c\u2018(\[]?[A-Z0-9])""")
_DOT = "\u2024"                                               # placeholder for protected dots


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Newlines are always boundaries (reference highlights use them)."""
    if not text or not text.strip():
        return []
    protected = _DOTTED.sub(lambda m: m.group(0).replace(".", _DOT), text)
    protected = _ABBR.sub(lambda m: m.group(1) + _DOT, protected)
    protected = _INITIAL.sub(lambda m: m.group(1) + _DOT, protected)
    protected = _DECIMAL.sub(_DOT, protected)
    protected = _BOUNDARY.sub(lambda m: m.group(1) + "\n", protected)
    parts = [p.replace(_DOT, ".").strip() for p in re.split(r"\n+", protected)]
    return [_WHITESPACE.sub(" ", p) for p in parts if p]


# --------------------------------------------------------------------------------------------
# Token budgets: truncation and chunking
# --------------------------------------------------------------------------------------------
def fit_sentences(sentences: list[str], lengths: list[int], budget: int) -> str:
    """Longest prefix of whole sentences whose token count fits ``budget``.

    The first sentence is always kept (the tokenizer hard-truncates it later if it is too long).
    """
    if not sentences:
        return ""
    used, kept = 0, []
    for sent, n in zip(sentences, lengths):
        if kept and used + n > budget:
            break
        kept.append(sent)
        used += n
    return " ".join(kept)


def chunk_sentences(sentences: list[str], lengths: list[int], budget: int) -> list[str]:
    """Greedily group consecutive sentences into chunks of at most ``budget`` tokens.

    A single sentence longer than the budget becomes its own chunk (hard-truncated later).
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for sent, n in zip(sentences, lengths):
        if current and used + n > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(sent)
        used += n
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


def word_count(text: str) -> int:
    return len(text.split())
