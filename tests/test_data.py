import pandas as pd
import pytest

from ml import data


def _long(s: str) -> str:
    return (s + " ") * 25  # > 20 words


def test_prepare_removes_exact_and_prefix_duplicates():
    raw = pd.DataFrame({
        "raw_text": [_long("alpha story"), _long("alpha story"), _long("beta story"),
                     _long("gamma story")],
        "label": [1, 1, 0, 0],
    })
    df, stats = data.prepare(raw, min_words=20)
    assert len(df) == 3
    assert stats["rows_after_dedup"] == 3


def test_prepare_drops_label_conflicts():
    raw = pd.DataFrame({"raw_text": [_long("same text"), _long("same text"), _long("other")],
                        "label": [1, 0, 0]})
    df, _ = data.prepare(raw, min_words=20)
    assert list(df["clean_text"].str.startswith("other")) == [True]


def test_prepare_drops_short_rows():
    raw = pd.DataFrame({"raw_text": ["too short", _long("long enough")], "label": [1, 0]})
    df, _ = data.prepare(raw, min_words=20)
    assert len(df) == 1


def test_split_is_stratified_disjoint_and_reproducible():
    df = pd.DataFrame({"clean_text": [f"article number {i} " * 30 for i in range(200)],
                       "label": [i % 2 for i in range(200)]})
    a = data.split(df, seed=1)
    b = data.split(df, seed=1)
    data.assert_no_overlap(*a)
    assert [x["clean_text"].tolist() for x in a] == [x["clean_text"].tolist() for x in b]
    assert abs(a[2]["label"].mean() - 0.5) < 0.05


def test_assert_no_overlap_detects_leakage():
    f1 = pd.DataFrame({"clean_text": ["x", "y"]})
    f2 = pd.DataFrame({"clean_text": ["y", "z"]})
    with pytest.raises(AssertionError):
        data.assert_no_overlap(f1, f2)


def test_load_isot_missing_files_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset preparation"):
        data.load_isot(tmp_path)


def test_load_isot_validates_columns(tmp_path):
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "True.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        data.load_isot(tmp_path)
