import pandas as pd
import pytest

from text_summarization.data import dataset_stats, find_split_files, load_dataset, sample_rows


def test_load_sample_is_cleaned(sample_dir):
    df = load_dataset(sample_dir / "test.csv")
    assert list(df.columns) == ["id", "article", "reference"] and len(df) == 40
    assert not df["article"].str.contains(r"\(CNN\)|PUBLISHED:|UPDATED:").any()
    assert df["reference"].str.contains("\n").all()               # one highlight per line is preserved
    assert df["id"].map(type).eq(str).all()


def test_find_split_files(sample_dir, tmp_path):
    assert set(find_split_files(sample_dir)) == {"train", "validation", "test"}
    (tmp_path / "val.csv").write_text("a,b\n", encoding="utf-8")
    assert find_split_files(tmp_path)["validation"].name == "val.csv"


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_dataset(tmp_path / "nope.csv")


def test_missing_column_message(sample_dir):
    with pytest.raises(KeyError, match="--text-col"):
        load_dataset(sample_dir / "test.csv", text_col="body")


def test_custom_columns_without_id(tmp_path):
    f = tmp_path / "c.csv"
    pd.DataFrame({"body": ["This is a long enough article text here."], "abstract": ["Short summary here"]}).to_csv(
        f, index=False)
    df = load_dataset(f, text_col="body", summary_col="abstract", id_col=None)
    assert df.loc[0, "id"] == "0" and df.loc[0, "reference"] == "Short summary here"


def test_short_and_empty_rows_dropped(tmp_path):
    f = tmp_path / "d.csv"
    pd.DataFrame({"article": ["ok " * 10, "tiny", None, "ok " * 10], "highlights": ["good summary", "x", "y", None]}
                 ).to_csv(f, index=False)
    assert len(load_dataset(f)) == 1
    with pytest.raises(ValueError, match="No usable rows"):
        load_dataset(f, min_article_words=500)


def test_nrows_and_no_clean(sample_dir):
    assert len(load_dataset(sample_dir / "train.csv", nrows=10)) == 10
    raw = load_dataset(sample_dir / "train.csv", clean=False)
    assert raw["article"].str.contains(r"\(CNN\)|PUBLISHED:").any()


def test_sample_rows_reproducible(sample_dir):
    df = load_dataset(sample_dir / "train.csv")
    a, b = sample_rows(df, 10, seed=1), sample_rows(df, 10, seed=1)
    assert a.equals(b) and len(a) == 10 and not a.equals(sample_rows(df, 10, seed=2))
    assert len(sample_rows(df, None)) == len(df) and len(sample_rows(df, 10_000)) == len(df)


def test_dataset_stats(sample_dir):
    stats = dataset_stats(load_dataset(sample_dir / "test.csv"))
    assert stats["rows"] == 40 and 0 < stats["compression_mean"] < 1
    assert stats["article_words_p95"] >= stats["article_words_median"]
