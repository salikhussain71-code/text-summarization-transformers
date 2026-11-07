# Data

Datasets are **not committed** to Git (see `.gitignore`). Follow the steps below.

## Dataset: CNN / DailyMail news summarisation (Kaggle)

* Kaggle page: <https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail>
* News articles from CNN and the Daily Mail, each with human-written **highlights** (bullet-point summaries)
* Files: `train.csv` (about 287k articles), `validation.csv` (about 13k), `test.csv` (about 11k)
* Columns: `id`, `article`, `highlights`

### Option A - manual download
1. Open the Kaggle link and click **Download**.
2. Unzip and place the three files here:
   ```
   data/raw/train.csv
   data/raw/validation.csv
   data/raw/test.csv
   ```

### Option B - Kaggle CLI
```bash
pip install kaggle                       # put your API token in ~/.kaggle/kaggle.json
kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail -p data/raw --unzip
```
The files are large (the training file is more than 1 GB). Evaluation only needs `test.csv`; fine-tuning
reads at most the first 50,000 rows of `train.csv`.

## What the pipeline does to the data (`preprocessing.py`, `data.py`)

| Step | Why |
|---|---|
| Remove `LONDON, England (CNN) --` datelines | The model should summarise content, not agency boilerplate |
| Remove Daily Mail `By . Reporter . PUBLISHED: ... UPDATED: ...` headers and `Last updated at ...` lines | Same |
| Collapse whitespace | Cleaner token counts |
| Reference highlights: one highlight per line | Needed for ROUGE-Lsum (summary-level ROUGE) |
| Drop rows with an empty / very short article or summary | Avoid degenerate scores |

These are regular-expression heuristics written from the known formats of the corpus; if your copy looks
different, inspect `reports/examples.md` and adjust `clean_article`.

## Using your own data (also used for fine-tuning)

Any CSV with an article column and a summary column works:

```bash
python -m text_summarization.evaluate --data my_news.csv --text-col body --summary-col abstract --id-col ""
python -m text_summarization.finetune --train-data my_news.csv --text-col body --summary-col abstract
```

## Models are downloaded separately

Pre-trained models (`distilbart-cnn-12-6`, `t5-small`, `bart-large-cnn`, ...) are downloaded from the Hugging Face
Hub the first time they are used and cached in `~/.cache/huggingface`. This needs an internet connection and, for the
larger models, 1-2 GB of disk space.

## Bundled sample (`data/sample/`)

`train.csv` (120 rows), `validation.csv` (30) and `test.csv` (40) are **synthetic**, template-generated articles in
the Kaggle format, created by `scripts/generate_sample_data.py`. They exist only for unit tests, CI and the offline
smoke test (`make smoke`). **Never report results obtained on them.**

## Folders

| Path | Purpose | Tracked by Git |
|---|---|---|
| `data/raw/` | Original Kaggle downloads | no |
| `data/processed/` | Reserved for derived data | no |
| `data/sample/` | Synthetic smoke-test data | yes |
