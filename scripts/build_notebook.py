"""Builds notebooks/01_text_summarization.ipynb (kept as a script so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s.strip())  # noqa: E731
code = lambda s: nbf.v4.new_code_cell(s.strip())  # noqa: E731

nb.cells = [
md("""
# Text Summarization with Pre-trained Transformers
**Task 7 - NLP Level 3** | Dataset: CNN / DailyMail (Kaggle) | Models: BART / T5 / Pegasus (encoder-decoder) | Metric: ROUGE

The reusable code lives in `src/text_summarization/`; `python -m text_summarization.evaluate` runs the full benchmark.
This notebook walks through the ideas:

1) load & clean -> 2) explore lengths and model input limits -> 3) extractive baselines (Lead-3, TextRank) ->
4) abstractive model -> 5) ROUGE comparison -> 6) qualitative examples -> 7) long documents: truncate vs chunk -> 8) fine-tuning (optional)
"""),
code("""
import sys
import tempfile
from pathlib import Path

try:
    import text_summarization  # noqa: F401  (works after `pip install -e .`)
except ImportError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
from IPython.display import Image, display

from text_summarization import config, visualize
from text_summarization.data import dataset_stats, find_split_files, load_dataset, sample_rows
from text_summarization.extractive import LeadSummarizer, TextRankSummarizer
from text_summarization.preprocessing import clean_article, split_sentences, word_count
from text_summarization.rouge import ROUGE_TYPES, aggregate, score_pairs, summary_stats
from text_summarization.summarizer import AbstractiveSummarizer, ModelLoadError
from text_summarization.testing import build_tiny_seq2seq

pd.set_option("display.max_colwidth", 160)
FIG_DIR = Path(tempfile.mkdtemp())   # notebook figures are temporary; the CLI saves the final ones
N = 30                               # number of test articles (raise it on a real run)
"""),
md("## 1. Load and clean the data"),
code("""
files = find_split_files(config.RAW_DIR)
if "test" in files:
    data_dir = config.RAW_DIR
else:
    print("CNN-DailyMail not found in data/raw/ - using the SYNTHETIC sample (results are NOT meaningful).")
    print("See data/README.md to download the real dataset.")
    data_dir = config.SAMPLE_DIR
df = sample_rows(load_dataset(find_split_files(data_dir)["test"]), N)
articles, references = df["article"].tolist(), df["reference"].tolist()
print(len(df), "articles |", {k: round(v, 1) for k, v in dataset_stats(df).items() if "mean" in k})
df.loc[0, ["article", "reference"]]
"""),
md("Boilerplate such as `LONDON, England (CNN) --` or the Daily Mail `PUBLISHED / UPDATED` header is removed so the model summarises content:"),
code("""
print(clean_article("LONDON, England (CNN) -- The bishop said on Tuesday that taxes would rise."))
print(split_sentences('Dr. Smith met Mr. Jones in the U.S. on Jan. 5. The price rose 3.5 percent.'))
"""),
md("## 2. Explore lengths"),
code("""
display(Image(str(visualize.plot_dataset_overview(df["article"].map(word_count), df["reference"].map(word_count), FIG_DIR))))
"""),
md("## 3. Extractive baselines (bonus): Lead-3 and TextRank"),
code("""
baselines = [LeadSummarizer(3), TextRankSummarizer(3)]
predictions = {b.name: b.summarize(articles) for b in baselines}
print("Lead-3  :", predictions["Lead-3"][0][:300])
print("TextRank:", predictions["TextRank-3"][0][:300])
"""),
md("## 4. Abstractive model (encoder-decoder)\\nPresets: `distilbart-cnn-12-6`, `bart-large-cnn`, `t5-small`, `t5-base`, `pegasus-cnn_dailymail`. The first use downloads the model from the Hugging Face Hub; **without internet this notebook falls back to a tiny random model** just to demonstrate the mechanics."),
code("""
MODEL = "t5-small"
try:
    summarizer = AbstractiveSummarizer(MODEL, max_summary_tokens=120, min_summary_tokens=30)
except ModelLoadError as err:
    print("Could not download", MODEL, "- using a tiny RANDOM model (its text is gibberish).")
    MODEL = "tiny-t5"
    summarizer = AbstractiveSummarizer(str(build_tiny_seq2seq(tempfile.mkdtemp(), "t5")), num_beams=1,
                                       max_summary_tokens=24, min_summary_tokens=4)
print(summarizer.info())
print("Articles longer than the input limit:", f"{summarizer.truncation_rate(articles):.0%}")
"""),
code("""
predictions[summarizer.name] = summarizer.summarize(articles, batch_size=4)
predictions[summarizer.name][0]
"""),
md("## 5. Evaluate with ROUGE\\nF1 x 100, with a bootstrap 95% confidence interval for ROUGE-L. Small samples give wide intervals - differences of a point or two are not meaningful."),
code("""
rows = []
for name, preds in predictions.items():
    agg = aggregate(score_pairs(preds, references), n_boot=200)
    stats = summary_stats(preds, articles, references)
    rows.append({"system": name, **{t: agg[t] for t in ROUGE_TYPES},
                 "rougeL_ci": f"{agg['rougeL_ci_low']:.1f}-{agg['rougeL_ci_high']:.1f}",
                 "words": stats["summary_words"], "novel_bigrams": stats["novel_bigrams"]})
results = pd.DataFrame(rows).sort_values("rougeL", ascending=False).reset_index(drop=True)
results.round(2)
"""),
code("""
display(Image(str(visualize.plot_rouge_comparison(results, FIG_DIR))))
"""),
md("## 6. Read some summaries\\nROUGE cannot detect hallucinations - always look at examples."),
code("""
i = 0
print("ARTICLE  :", articles[i][:500], "...")
print("\\nREFERENCE:", references[i].replace("\\n", " | "))
for name, preds in predictions.items():
    print(f"\\n{name}:", preds[i])
"""),
md("## 7. Long documents: truncate or chunk?\\nModels read a limited number of tokens. `truncate` ignores the rest; `chunk` summarises each chunk and then the summaries."),
code("""
long_article = " ".join(articles[:6])
for mode in ("truncate", "chunk"):
    s = AbstractiveSummarizer(summarizer.source, long_doc=mode, truncation="sentences", max_input_tokens=summarizer.max_input_tokens,
                              num_beams=1, max_summary_tokens=summarizer.generation["max_length"],
                              min_summary_tokens=summarizer.generation["min_length"])
    print(f"{mode:8s} | input tokens {s.token_lengths([long_article])[0]} (limit {s.max_input_tokens}) ->",
          s.summarize([long_article])[0][:200])
"""),
md("""
## 8. Fine-tuning (optional)
Fine-tuning needs a GPU for anything beyond toy sizes, so it is not run here. From a terminal:

```bash
python -m text_summarization.finetune --model t5-small --n-train 2000 --epochs 1
python -m text_summarization.evaluate --models models/t5-small-finetuned --n 100
```
It prints validation ROUGE before and after fine-tuning and saves the best epoch.
"""),
md("""
## Conclusions
* Lead-3 is a strong baseline for news; abstractive models are judged on whether they beat it on ROUGE-2 / ROUGE-L *and* read well.
* Input limits matter: check how many articles exceed the model's limit and consider sentence-aware truncation or chunking.
* ROUGE says nothing about factual accuracy - read examples before trusting a system.
* **Next steps:** run the benchmark on 200+ real articles with several models, then fine-tune on your own data.

> If you ran this on the synthetic sample or with the tiny fallback model, repeat it with the real dataset and models before drawing conclusions.
"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/01_text_summarization.ipynb")
print("notebook written")
