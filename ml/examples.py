"""Dashboard examples taken from YOUR dataset, restricted to the held-out TEST split.

Why only the test split: the model trained on ~70% of the articles, so showing it those would prove
nothing. The test split was never used for fitting, calibration or thresholds.

How the split is recovered without storing it: training is deterministic (same files, same cleaning,
same seed), so the split is rebuilt, and then PROVEN by re-computing the recorded test accuracy with
the active model. If the dataset files changed since training, or the numbers do not match exactly,
nothing is written.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml import data as data_mod
from ml.inference.predictor import Predictor
from ml.preprocessing import word_count


class ExampleError(RuntimeError):
    """The held-out split could not be reconstructed and verified."""


def reconstruct_splits(predictor: Predictor, data_dir: str | Path):
    """Return (train, val, test, test_accuracy). `test` has raw_text, title and model_p_fake."""
    meta = predictor.metadata
    ds = meta["dataset"]
    if ds.get("type") != "isot":
        raise ExampleError("Examples are only supported for models trained with --dataset isot.")
    for name, recorded in (ds.get("sha256") or {}).items():
        path = Path(data_dir) / name
        if not path.exists():
            raise ExampleError(
                f"{path} not found. Point --data-dir at the folder used for training.")
        if data_mod.file_sha256(path) != recorded:
            raise ExampleError(
                f"{name} is not the same file the model was trained on, so held-out examples "
                "cannot be guaranteed. Restore the original file or retrain.")
    cfg = meta["config"]
    raw = data_mod.load_isot(data_dir)
    df, _ = data_mod.prepare(raw, cfg["data"]["min_words"], cfg["data"]["remove_source_leaks"],
                             keep_raw=True)
    train, val, test = data_mod.split(df, meta["seed"], cfg["data"]["val_size"],
                                      cfg["data"]["test_size"])
    p = predictor.classifier.predict_proba(test["clean_text"].tolist())[:, 1]
    accuracy = float(((p >= 0.5).astype(int) == test["label"].to_numpy()).mean())
    recorded_acc = meta["test_metrics"]["accuracy"]
    if abs(accuracy - recorded_acc) > 1e-9:
        raise ExampleError(
            f"Rebuilt test split gives accuracy {accuracy:.6f} but training recorded "
            f"{recorded_acc:.6f}. The split could not be reproduced, so no examples were written.")
    return train, val, test.assign(model_p_fake=p), accuracy


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."


def build_examples(test: pd.DataFrame, per_class: int = 8, wrong_per_class: int = 1, seed: int = 7,
                   min_words: int = 40, max_words: int = 700, max_chars: int = 6000) -> list[dict]:
    """Pick readable examples per class. A few cases the model gets WRONG are mixed in on purpose
    (unlabelled), so the dashboard can show where it fails, not just where it shines."""
    rng = np.random.default_rng(seed)
    t = test.copy()
    t["words"] = t["raw_text"].map(word_count)
    t["wrong"] = (t["model_p_fake"] >= 0.5).astype(int) != t["label"]
    t = t[(t["words"] >= min_words) & (t["words"] <= max_words)]
    out: list[dict] = []
    for label, name in ((data_mod.LABEL_REAL, "real"), (data_mod.LABEL_FAKE, "fake")):
        pool = t[t["label"] == label]
        wrong, right = pool[pool["wrong"]], pool[~pool["wrong"]]
        n_wrong = min(wrong_per_class, len(wrong), per_class)
        picks = []
        if n_wrong:
            picks.append(wrong.sample(n_wrong, random_state=int(rng.integers(1_000_000_000))))
        picks.append(right.sample(min(per_class - n_wrong, len(right)),
                                  random_state=int(rng.integers(1_000_000_000))))
        for idx, row in pd.concat(picks).iterrows():
            title = str(row.get("title") or "").strip() if "title" in row else ""
            out.append({"id": f"{name}-{idx}", "label": name,
                        "title": title[:120] or _clip(row["raw_text"], 100),
                        "text": _clip(row["raw_text"], max_chars)})
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def write_examples(
    path: str | Path, examples: list[dict], model_version: str, accuracy: float
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": model_version,
        "source": "held-out test split of the active model (never used for training)",
        "split_verified_by_test_accuracy": accuracy,
        "examples": examples,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path