.PHONY: install install-dev evaluate evaluate-fast finetune summarize tiny-model smoke test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Benchmark baselines + abstractive models on 100 test articles (downloads models on first run)
evaluate:
	python -m text_summarization.evaluate --n 100

# Baselines only (no downloads, no GPU) - Lead-3 and TextRank
evaluate-fast:
	python -m text_summarization.evaluate --n 200 --models

# Fine-tune t5-small on 2,000 training articles (see --help for all options)
finetune:
	python -m text_summarization.finetune --model t5-small --n-train 2000 --epochs 1

summarize:
	python -m text_summarization.summarize --method textrank "Paste an article here. It needs a few sentences. Only then can the summariser choose the most central ones. The rest is dropped."

# Build a tiny random-weight T5 so the whole pipeline can be tried OFFLINE (output is gibberish)
tiny-model:
	python -m text_summarization.testing --family t5 --out models/tiny-t5

# Offline end-to-end smoke test on the synthetic sample + tiny model
smoke: tiny-model
	python -m text_summarization.evaluate --data-dir data/sample --n 10 --models models/tiny-t5 --max-summary-tokens 24 --min-summary-tokens 4 --no-update-readme

test:
	pytest

lint:
	ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
