# Methodology

## 1. Task
Generate a **concise summary** of a news article. Two families of methods are covered:

* **Abstractive** - a pre-trained *encoder-decoder* (sequence-to-sequence) Transformer reads the article and
  *writes* a new summary: BART, T5 or PEGASUS.
* **Extractive** (bonus) - *select* the most important existing sentences: Lead-3 and TextRank.

## 2. Pipeline
```
CSV -> clean article -> [truncate / chunk to the model's input limit] -> encoder-decoder -> summary
                                                                                          |
                              reference highlights ------------------------------------> ROUGE
```

## 3. Input length: truncation and long documents
Transformers can only read a fixed number of tokens at once.

| Model family | Encoder limit used | Notes |
|---|---|---|
| BART / DistilBART | 1,024 tokens | learned absolute positions |
| PEGASUS | 1,024 tokens | |
| T5 | 512 tokens | relative positions; trained with 512 |

A typical CNN/DM article has roughly 700-800 tokens on average with a long tail of much longer ones, so **a noticeable share of articles exceeds the 512-token T5 limit and a smaller share exceeds 1,024**. The evaluation reports this share per model (`truncated_share`). Two strategies are implemented (`summarizer.py`, `preprocessing.py`):

| Option | Behaviour |
|---|---|
| `--truncation tokens` (default) | Cut at the token limit. Simple; may end mid-sentence. |
| `--truncation sentences` | Keep only **whole sentences** that fit. The model never sees half a sentence. |
| `--long-doc truncate` (default) | Ignore everything beyond the limit. For news this is usually fine because the key facts come first. |
| `--long-doc chunk` | Split an over-long document into budget-sized chunks of sentences, summarise each chunk, then summarise the concatenated chunk summaries (applied recursively, at most twice). Preserves coverage of long documents but loses cross-chunk coherence and is slower. |

The T5 task prefix `summarize: ` is added automatically and counted against the budget, as are the special
tokens.

## 4. Generation settings
Beam search with the settings published for each checkpoint (`presets.py`), e.g. for BART-CNN:
`num_beams=4`, `min_length=56`, `max_length=142`, `length_penalty=2.0`, `no_repeat_ngram_size=3`.
Every setting can be overridden on the command line (`--num-beams`, `--max-summary-tokens`, ...).
Batches are sorted by length for speed and results are returned in input order.

## 5. Extractive baselines (`extractive.py`)
* **Lead-N** - the first N sentences. News articles put the key facts first, so Lead-3 is a very strong baseline
  that many abstractive systems barely beat.
* **TextRank** - sentences are graph nodes, edges are TF-IDF cosine similarities, PageRank scores the nodes and the
  top N sentences are returned in their original order (NumPy only). `textrank-lead` adds a mild preference for early
  sentences.

## 6. Evaluation (`rouge.py`)
**ROUGE** compares a generated summary with the reference by n-gram overlap (Google's `rouge-score`, Porter
stemming on):

| Metric | Meaning |
|---|---|
| ROUGE-1 / ROUGE-2 | overlap of unigrams / bigrams |
| ROUGE-L | longest common subsequence |
| ROUGE-Lsum | ROUGE-L computed per sentence and combined (summary level; the usual headline number for CNN/DM) |

All scores are F1 x 100. Additionally:

* **Bootstrap 95 % confidence intervals** (1,000 resamples over articles) - with a small evaluation sample,
  differences of one or two ROUGE points are usually **not** significant.
* **Novel n-grams** - the share of summary bigrams that never occur in the article (0 % = pure copy-paste). It shows
  how abstractive a system really is.
* Length (words) and speed (seconds per article).
* All systems see the **same random sample** of articles (fixed seed), and per-article predictions are saved
  (`reports/predictions/`) together with qualitative examples (`reports/examples.md`).

## 7. Fine-tuning (bonus, `finetune.py`)
Plain PyTorch loop: AdamW, linear warm-up and decay, gradient accumulation and clipping, optional bfloat16 autocast.
ROUGE and loss on a held-out validation subset are measured **before** fine-tuning (epoch 0) and after every epoch;
the best epoch (highest ROUGE-L, ties broken by validation loss) is saved with a `summarizer_meta.json`
(prefix, input limit, generation settings) so it loads like any other model:

```bash
python -m text_summarization.finetune --model t5-small --n-train 2000 --epochs 1
python -m text_summarization.evaluate --models models/t5-small-finetuned
```
It works on your own article/summary CSV (`--train-data --text-col --summary-col`). Use a GPU for anything beyond a
few hundred examples.

## 8. Published reference points
Papers report, on the **full** CNN/DM test set, roughly: Lead-3 ROUGE-1/2/L about 40.3 / 17.7 / 36.6 (See et al.,
2017) and BART-large about 44.2 / 21.3 / 40.9 (Lewis et al., 2020). These numbers are given only for orientation:
they come from other ROUGE implementations / settings, use the whole test set and differ from a small sample scored
with this repository. Compare systems with **each other** inside your own run.

## 9. Limitations and responsible use
* **Hallucination**: abstractive models can state things the article does not say. ROUGE does not measure factual
  consistency - always read samples (`reports/examples.md`).
* ROUGE rewards word overlap, so it under-rates good paraphrases and over-rates copying (note the strong Lead-3).
* Models are English-news specific; other domains or languages need other checkpoints or fine-tuning.
* `t5-small` was multi-task pre-trained (including CNN/DM) but is not specialised for it, so it is usually weaker
  than the CNN/DM-fine-tuned BART models.
* Chunked summarisation of very long documents may drop cross-section context.
* Do not use generated summaries of sensitive material (legal, medical, financial) without human review.

## 10. Extensions
Longformer / LED for long inputs, factual-consistency metrics (e.g. QA-based or entailment-based), BERTScore,
LoRA fine-tuning of larger models, query-focused summaries, a Streamlit demo.
