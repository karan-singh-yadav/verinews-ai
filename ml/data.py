"""Dataset loading, de-duplication and leakage-safe splitting.

Label convention everywhere in this project: 1 = fake, 0 = real.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.preprocessing import clean_text, word_count

LABEL_FAKE = 1
LABEL_REAL = 0
LABEL_NAMES = {LABEL_REAL: "real", LABEL_FAKE: "fake"}


def build_text(title: object, body: object) -> str:
    title = "" if title is None or (isinstance(title, float) and title != title) else str(title)
    body = "" if body is None or (isinstance(body, float) and body != body) else str(body)
    return f"{title}. {body}".strip(". ").strip()


def load_isot(raw_dir: str | Path) -> pd.DataFrame:
    """Load the 'Fake and Real News Dataset' (Fake.csv + True.csv; columns title,text,subject,date).

    `subject` and `date` are deliberately dropped: subject categories differ systematically
    between classes in this dataset and would leak the label.
    """
    raw_dir = Path(raw_dir)
    fake_path, true_path = raw_dir / "Fake.csv", raw_dir / "True.csv"
    for p in (fake_path, true_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. See README 'Dataset preparation' for the download step."
            )
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)
    for name, frame in (("Fake.csv", fake), ("True.csv", real)):
        missing = {"title", "text"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    fake["label"] = LABEL_FAKE
    real["label"] = LABEL_REAL
    df = pd.concat([fake, real], ignore_index=True)
    df["raw_text"] = [build_text(t, b) for t, b in zip(df["title"], df["text"], strict=True)]
    df["title"] = df["title"].fillna("").astype(str)
    return df[["raw_text", "label", "title"]]


def load_generic_csv(
    path: str | Path, text_col: str, label_col: str, fake_values: tuple = (1, "1", "fake", "FAKE")
) -> pd.DataFrame:
    """Load any single CSV with a text column and a label column."""
    df = pd.read_csv(path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "raw_text": df[text_col].astype(str),
            "label": df[label_col].isin(fake_values).astype(int),
        }
    )
    return out


def prepare(
    df: pd.DataFrame, min_words: int = 20, remove_source_leaks: bool = True, keep_raw: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Clean text, drop short rows, and remove duplicates / label conflicts.

    Duplicate handling (done BEFORE splitting so no article can sit in two splits):
      1. exact duplicates of the cleaned text
      2. near-duplicates sharing the same first 300 characters (re-posted articles)
      3. any group that carries conflicting labels is dropped entirely (ambiguous)
    """
    stats = {"rows_raw": int(len(df))}
    df = df.copy()
    df["clean_text"] = [clean_text(t, remove_source_leaks) for t in df["raw_text"]]
    df = df[df["clean_text"].map(word_count) >= min_words]
    stats["rows_after_min_words"] = int(len(df))

    df["_exact"] = df["clean_text"].map(lambda t: hashlib.md5(t.encode()).hexdigest())
    df["_prefix"] = df["clean_text"].str[:300].map(lambda t: hashlib.md5(t.encode()).hexdigest())

    for key in ("_exact", "_prefix"):
        conflicting = df.groupby(key)["label"].nunique()
        conflicting = conflicting[conflicting > 1].index
        stats[f"conflicting_groups_{key.strip('_')}"] = int(len(conflicting))
        df = df[~df[key].isin(conflicting)]
        df = df.drop_duplicates(subset=key, keep="first")

    stats["rows_after_dedup"] = int(len(df))
    stats["class_counts"] = {
        LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()
    }
    cols = ["clean_text", "label"]
    if keep_raw:  # used only to show original text in the dashboard examples; training ignores it
        cols += [c for c in ("raw_text", "title") if c in df.columns]
    return df[cols].reset_index(drop=True), stats


def split(
    df: pd.DataFrame, seed: int = 42, val_size: float = 0.15, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def assert_no_overlap(*frames: pd.DataFrame, col: str = "clean_text") -> None:
    """Raise if any article text appears in more than one split (leakage guard)."""
    seen: set[str] = set()
    for i, frame in enumerate(frames):
        current = set(frame[col])
        overlap = seen & current
        if overlap:
            raise AssertionError(f"{len(overlap)} duplicated texts found across splits (split {i})")
        seen |= current


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
