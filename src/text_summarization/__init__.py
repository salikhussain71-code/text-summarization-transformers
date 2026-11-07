"""Text summarisation with pre-trained encoder-decoder models (BART / T5 / Pegasus).

* ``summarizer``   - abstractive summariser with token-aware truncation and long-document chunking
* ``extractive``   - Lead-N and TextRank baselines
* ``rouge``        - ROUGE evaluation with bootstrap confidence intervals and abstractiveness stats
* ``finetune``     - fine-tune a pre-trained summariser on your own data
"""

__version__ = "1.0.0"
