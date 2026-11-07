# Text Summarization with Pre-trained Transformers (BART / T5 / Pegasus)

[![CI](https://github.com/salikhussain71-code/text-summarization-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/salikhussain71-code/text-summarization-transformers/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.40%2B-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A complete news-summarisation project: **abstractive** summaries from pre-trained **encoder-decoder** models
(**BART, T5, Pegasus**), **extractive** baselines (**Lead-3, TextRank**), careful handling of **model input limits**,
**ROUGE** evaluation with confidence intervals, and **fine-tuning** on your own data.

> **Task 7 - Level 3 (NLP Tasks).** Dataset: CNN / DailyMail (Kaggle). Includes both bonus items:
> extractive summarisation with TextRank and fine-tuning a pre-trained summariser on a custom dataset.

---

## Table of contents
1. [Highlights](#highlights)
2. [Task requirements checklist](#task-requirements-checklist)
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [Usage](#usage)
6. [Results](#results)
7. [Methodology](#methodology)
8. [Testing & code quality](#testing--code-quality)
9. [Limitations & responsible use](#limitations--responsible-use)
10. [License](#license)

## Highlights
* **Any encoder-decoder summariser**: presets for `distilbart-cnn-12-6`, `bart-large-cnn`, `t5-small`, `t5-base`,
  `pegasus-cnn_dailymail`, plus any Hugging Face model id or local folder. T5 prefixes and published generation
  settings are applied automatically.
* **Input-limit handling**: token truncation, **sentence-aware truncation**, and a **long-document mode** that
  summarises chunks and then merges them; the share of articles that exceed each model's limit is reported.
* **Rigorous ROUGE**: ROUGE-1/2/L/Lsum with Porter stemming, **bootstrap 95 % confidence intervals**, same random
  article sample for every system, per-article predictions saved.
* **Beyond ROUGE**: abstractiveness (novel n-grams), length, speed, and a side-by-side examples page.
* **Strong baselines**: Lead-3 and a NumPy TextRank, so you can see what the neural models add.
* **Fine-tuning** loop (AdamW, warm-up, gradient accumulation, bf16) with before/after ROUGE; the result loads like any
  other model.
* **Works offline for testing**: a tiny random T5/BART builder lets the whole pipeline run without downloads
  (65 tests, CI-friendly).

## Task requirements checklist

| Requirement | Where it is implemented |
|---|---|
| Dataset: CNN-DailyMail (Kaggle) | `data/README.md`, `src/text_summarization/data.py` |
| Generate concise summaries from long documents using NLP models | `summarizer.py` (`AbstractiveSummarizer`), `summarize.py` CLI |
| Encoder-decoder architecture (T5, BART or Pegasus) | `presets.py`, `summarizer.py` |
| Preprocess long texts and truncate to model input limits | `preprocessing.py`, `summarizer.py` (`--truncation`, `--long-doc`) |
| Evaluate summary quality using ROUGE | `rouge.py`, `evaluate.py` |
| Tools: ROUGE, Hugging Face Transformers, Pandas | `rouge-score`, `transformers`, `pandas` |
| **Bonus:** extractive summarisation (TextRank) | `extractive.py` (TextRank + Lead-N) |
| **Bonus:** fine-tune a pre-trained summariser on a custom dataset | `finetune.py` |

## Project structure

```
text-summarization-transformers/
├── README.md
├── LICENSE
├── Makefile                      # shortcuts: make evaluate / finetune / smoke / test / lint
├── pyproject.toml                # package metadata, pytest & ruff config
├── requirements.txt              # runtime dependencies
├── requirements-dev.txt          # + pytest, ruff, jupyter
├── .gitignore
├── .github/workflows/ci.yml      # lint + tests on every push (CPU PyTorch, no downloads)
├── data/
│   ├── README.md                 # how to download CNN-DailyMail, cleaning steps
│   ├── raw/                      # put train.csv, validation.csv, test.csv here (git-ignored)
│   ├── processed/
│   └── sample/                   # synthetic CSVs for smoke tests only
├── docs/methodology.md           # input limits, generation, ROUGE, fine-tuning, limitations
├── notebooks/
│   └── 01_text_summarization.ipynb
├── scripts/                      # generate_sample_data.py, build_notebook.py
├── src/text_summarization/
│   ├── config.py                 # paths and constants
│   ├── presets.py                # BART / T5 / Pegasus presets and generation settings
│   ├── preprocessing.py          # cleaning, sentence splitting, truncation / chunking helpers
│   ├── data.py                   # CSV loading, sampling, statistics
│   ├── summarizer.py             # abstractive summariser (batching, truncation, long documents)
│   ├── extractive.py             # Lead-N and TextRank
│   ├── rouge.py                  # ROUGE, bootstrap CIs, abstractiveness
│   ├── visualize.py              # charts
│   ├── evaluate.py               # benchmark CLI
│   ├── finetune.py               # fine-tuning CLI
│   ├── summarize.py              # summarise-your-own-text CLI
│   └── testing.py                # tiny offline model builder
├── tests/                        # 65 pytest unit & integration tests
├── models/                       # fine-tuned checkpoints (git-ignored)
└── reports/
    ├── figures/                  # PNG charts
    ├── metrics/                  # CSV / JSON / Markdown results
    ├── predictions/              # per-system, per-article summaries and scores
    └── examples.md               # qualitative side-by-side examples
```

## Quick start

**Requirements:** Python 3.9+. A GPU is optional but makes the larger models much faster. Pre-trained models
(0.2-2 GB each) are downloaded from the Hugging Face Hub on first use, so an internet connection is needed.

```bash
# 1. Clone
git clone https://github.com/salikhussain71-code/text-summarization-transformers.git
cd text-summarization-transformers

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install (add the PyTorch build for your hardware first if you need a specific CUDA version: pytorch.org)
pip install -r requirements.txt
pip install -e .

# 4. Get the data (see data/README.md) -> data/raw/train.csv, validation.csv, test.csv
kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail -p data/raw --unzip

# 5. Benchmark Lead-3, TextRank, DistilBART and T5-small on 100 random test articles
python -m text_summarization.evaluate --n 100
```

No data yet? Try the pipeline on your own text right away:
`python -m text_summarization.summarize --method textrank "Paste an article here ..."`.

## Usage

### Evaluate
```bash
python -m text_summarization.evaluate --n 100                                  # baselines + distilbart + t5-small
python -m text_summarization.evaluate --models bart-large-cnn t5-base --n 200  # other models
python -m text_summarization.evaluate --models --n 0                           # extractive baselines on the full test set
python -m text_summarization.evaluate --truncation sentences --long-doc chunk  # input-length strategies
python -m text_summarization.evaluate --models models/t5-small-finetuned       # a model you fine-tuned
python -m text_summarization.evaluate --help                                   # all options
```
Outputs: `reports/metrics/rouge_results.{csv,md}`, `run_summary.json`, per-article predictions, figures, `reports/examples.md`
and the **Results** section below.

### Summarise your own text
```bash
python -m text_summarization.summarize --model t5-small "Long article text ..."
python -m text_summarization.summarize --file article.txt --method textrank --sentences 3
python -m text_summarization.summarize --csv news.csv --text-col content --out summaries.csv
python -m text_summarization.summarize --model bart-large-cnn --long-doc chunk --file long_report.txt
```
From Python:
```python
from text_summarization.summarizer import AbstractiveSummarizer

summarizer = AbstractiveSummarizer("distilbart-cnn-12-6", truncation="sentences", long_doc="chunk")
summarizer.summarize(["Article text ..."])
```

### Fine-tune on your own data (bonus)
```bash
python -m text_summarization.finetune --model t5-small --n-train 2000 --epochs 1            # CNN/DM subset
python -m text_summarization.finetune --model t5-small --train-data my_news.csv \
    --text-col body --summary-col abstract --epochs 3 --output models/t5-my-news           # your data
```
It reports validation ROUGE **before and after** fine-tuning and saves the best epoch.

### Offline smoke test (no downloads, no GPU)
```bash
make smoke      # builds a tiny random T5 and runs the whole evaluation on the synthetic sample
```
The tiny model's text is gibberish - this only proves the pipeline runs end to end.

### Notebook
```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/01_text_summarization.ipynb
```

## Results

<!-- RESULTS:START -->
_Run `python -m text_summarization.evaluate` on the CNN-DailyMail data - this section is then filled in automatically
with the ROUGE table, confidence intervals, truncation statistics and figures._
<!-- RESULTS:END -->

**What to expect:** Lead-3 is a very strong baseline on news; CNN/DM-fine-tuned BART-type models usually beat it on
ROUGE-2 and ROUGE-L, `t5-small` is typically weaker than the BART models, and TextRank usually lands below Lead-3.
With only 100 articles the confidence intervals are wide - increase `--n` before drawing conclusions.
`docs/methodology.md` lists published reference numbers for orientation (they are not directly comparable).

## Methodology

A short summary - see [`docs/methodology.md`](docs/methodology.md) for details.

1. **Clean** articles (CNN / Daily Mail boilerplate) and keep reference highlights one per line.
2. **Fit the input** to each model: token or sentence-aware truncation, or chunk-and-merge for long documents.
3. **Generate** with beam search using each checkpoint's published settings.
4. **Score** with ROUGE-1/2/L/Lsum plus bootstrap confidence intervals, novel n-grams, length and speed.
5. **Compare** with Lead-3 / TextRank, inspect qualitative examples, optionally **fine-tune** and re-evaluate.

## Testing & code quality

```bash
pip install -r requirements-dev.txt
pytest          # 65 tests (offline: tiny local T5 / BART models, nothing is downloaded)
ruff check .    # linting
```
GitHub Actions runs both on Python 3.10, 3.11 and 3.12 with CPU-only PyTorch.

## Limitations & responsible use
* Abstractive models can **hallucinate**; ROUGE does not measure factual consistency - read the examples.
* ROUGE favours word overlap, which is why Lead-3 is hard to beat.
* English news only; other domains need other checkpoints or fine-tuning.
* Very long documents are handled by truncation or chunking, both of which lose some context.
* Have a human review summaries of sensitive material.

## License
Released under the [MIT License](LICENSE). The CNN / DailyMail data and the pre-trained models have their own licences -
check the Kaggle and Hugging Face pages before redistributing anything.
