"""Training entry point:  python -m ml.training.train --dataset isot --data-dir data/raw

Pipeline
  load -> clean -> de-duplicate -> split(train/val/test)
  -> fit candidate models on TRAIN, compare on VALIDATION
  -> split VALIDATION in two: half calibrates probabilities, half picks thresholds
  -> evaluate the final calibrated model ONCE on TEST
  -> fit novelty detector on TRAIN, set its threshold from VALIDATION
  -> save artifacts + metadata, log everything to MLflow
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ml import data as data_mod
from ml.evaluation.metrics import calibration_report, classification_report_dict
from ml.inference.novelty import NoveltyDetector
from ml.preprocessing import word_count


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(cfg: dict) -> FeatureUnion:
    f = cfg["features"]
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=tuple(f["word_ngram_range"]),
                max_features=f["word_max_features"], min_df=f["min_df"], sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=tuple(f["char_ngram_range"]),
                max_features=f["char_max_features"], min_df=f["min_df"], sublinear_tf=True,
            )),
        ]
    )


def build_candidates(cfg: dict, seed: int) -> dict[str, Pipeline]:
    m = cfg["models"]
    cands = {
        "logistic_regression": LogisticRegression(
            C=m["logistic_regression"]["C"], max_iter=m["logistic_regression"]["max_iter"],
            random_state=seed,
        ),
        "linear_svm": LinearSVC(C=m["linear_svm"]["C"], random_state=seed),
    }
    if m["random_forest"].get("enabled"):
        cands["random_forest"] = RandomForestClassifier(
            n_estimators=m["random_forest"]["n_estimators"],
            max_depth=m["random_forest"]["max_depth"], n_jobs=-1, random_state=seed,
        )
    return {
        name: Pipeline([("features", build_features(cfg)), ("clf", clf)])
        for name, clf in cands.items()
    }


def raw_scores(model, texts: list[str]) -> np.ndarray:
    """Probabilities if available, otherwise decision values (ranking is enough for AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def select_confidence_threshold(
    y_true: np.ndarray, p_fake: np.ndarray, target: float, fallback: float
) -> tuple[float, dict]:
    """Lowest confidence threshold whose accepted predictions reach `target` accuracy (on val)."""
    pred = (p_fake >= 0.5).astype(int)
    conf = np.where(pred == 1, p_fake, 1 - p_fake)
    correct = pred == y_true
    best = None
    for t in np.unique(np.round(conf, 4)):
        mask = conf >= t
        if mask.sum() >= max(20, int(0.2 * len(conf))) and correct[mask].mean() >= target:
            best = float(t)
            break
    info = {"target_selective_accuracy": target, "selection_set_size": int(len(y_true))}
    if best is None:
        info["note"] = "target not reachable on validation data; fallback threshold used"
        return float(fallback), info
    mask = conf >= best
    info["coverage_at_threshold"] = float(mask.mean())
    info["selective_accuracy_at_threshold"] = float(correct[mask].mean())
    return best, info


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    np.random.seed(seed)

    # ---- data ----
    if args.dataset == "isot":
        raw = data_mod.load_isot(args.data_dir)
        data_fingerprint = {
            n: data_mod.file_sha256(Path(args.data_dir) / n) for n in ("Fake.csv", "True.csv")
        }
    elif args.dataset == "csv":
        raw = data_mod.load_generic_csv(args.csv_path, args.text_col, args.label_col)
        data_fingerprint = {Path(args.csv_path).name: data_mod.file_sha256(args.csv_path)}
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")

    df, prep_stats = data_mod.prepare(
        raw, cfg["data"]["min_words"], cfg["data"]["remove_source_leaks"]
    )
    train, val, test = data_mod.split(df, seed, cfg["data"]["val_size"], cfg["data"]["test_size"])
    data_mod.assert_no_overlap(train, val, test)
    if df["label"].nunique() < 2 or min(len(train), len(val), len(test)) < 10:
        raise ValueError("Dataset too small or single-class after cleaning; cannot train.")

    Xtr, ytr = train["clean_text"].tolist(), train["label"].to_numpy()
    Xva, yva = val["clean_text"].tolist(), val["label"].to_numpy()
    Xte, yte = test["clean_text"].tolist(), test["label"].to_numpy()

    # ---- candidate comparison on VALIDATION ----
    candidates = build_candidates(cfg, seed)
    comparison: dict[str, dict] = {}
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        s = raw_scores(pipe, Xva)
        thr = 0.5 if hasattr(pipe, "predict_proba") else 0.0
        comparison[name] = classification_report_dict(yva, (s >= thr).astype(int), s)
        fitted[name] = pipe
    metric = cfg["selection_metric"]
    best_name = max(comparison, key=lambda n: comparison[n][metric])
    best = fitted[best_name]

    # ---- calibration (val half A) and thresholds (val half B) ----
    idx = np.arange(len(Xva))
    cal_idx, thr_idx = train_test_split(idx, test_size=0.5, stratify=yva, random_state=seed)
    Xcal, ycal = [Xva[i] for i in cal_idx], yva[cal_idx]
    Xthr, ythr = [Xva[i] for i in thr_idx], yva[thr_idx]

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(best), method=cfg["calibration"]["method"]
    ).fit(Xcal, ycal)

    p_thr = calibrated.predict_proba(Xthr)[:, 1]
    conf_threshold, conf_info = select_confidence_threshold(
        ythr, p_thr, cfg["confidence"]["target_selective_accuracy"],
        cfg["confidence"]["fallback_threshold"],
    )

    # ---- novelty detector ----
    n = cfg["novelty"]
    novelty = NoveltyDetector(
        svd_components=n["svd_components"], max_features=n["max_features"],
        k_neighbors=n["k_neighbors"], max_reference_docs=n["max_reference_docs"], seed=seed,
    ).fit(Xtr)
    novelty.calibrate_threshold(Xva, n["threshold_percentile"])

    # ---- final evaluation on TEST (touched once) ----
    p_test = calibrated.predict_proba(Xte)[:, 1]
    test_report = classification_report_dict(yte, (p_test >= 0.5).astype(int), p_test)
    cal_report = calibration_report(yte, p_test)
    cal_report["valid"] = bool(cal_report["ece"] <= cfg["calibration"]["max_ece_for_valid"])
    test_nov = novelty.score(Xte)
    novelty_report = {
        "threshold": novelty.threshold,
        "percentile": n["threshold_percentile"],
        "test_fraction_flagged_ood": float(novelty.is_ood(test_nov).mean()),
        "note": "In-distribution test data is expected to be flagged at roughly "
                f"{100 - n['threshold_percentile']}%; this measures false-alarm rate, "
                "not OOD detection power.",
    }

    # ---- monitoring reference (derived numbers only, never article text) ----
    test_conf = np.maximum(p_test, 1 - p_test)
    ref_n = min(len(Xte), 5000)
    ref_idx = np.random.default_rng(seed).choice(len(Xte), ref_n, replace=False)
    reference = pd.DataFrame({
        "word_count": [word_count(Xte[i]) for i in ref_idx],
        "char_length": [len(Xte[i]) for i in ref_idx],
        "fake_probability": p_test[ref_idx],
        "confidence": test_conf[ref_idx],
        "novelty_score": test_nov[ref_idx],
    })
    monitoring_reference = {
        "source": "test split (never used for fitting, calibration or thresholds)",
        "n": int(len(Xte)),
        "fake_rate": float((p_test >= 0.5).mean()),
        "ood_rate": float(novelty.is_ood(test_nov).mean()),
        "low_confidence_rate": float((test_conf < conf_threshold).mean()),
    }

    # ---- persist ----
    version = args.version or datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, out_dir / "classifier.joblib")
    joblib.dump(novelty, out_dir / "novelty.joblib")

    metadata = {
        "model_version": version,
        "selected_model": best_name,
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "config": cfg,
        "dataset": {"type": args.dataset, "sha256": data_fingerprint, **prep_stats,
                    "split_sizes": {"train": len(train), "val": len(val), "test": len(test)}},
        "validation_comparison": comparison,
        "test_metrics": test_report,
        "calibration": cal_report,
        "confidence_threshold": conf_threshold,
        "confidence_threshold_info": conf_info,
        "novelty": novelty_report,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "git_sha": git_sha()},
        "monitoring_reference": monitoring_reference,
        "synthetic_data": bool(args.synthetic),
        "limitations": [
            "Labels reflect source-level labelling of one dataset, not fact-checked truth.",
            "Accuracy on the test split does not transfer to new topics, outlets or time periods.",
            "The model scores writing style and vocabulary; it cannot verify claims.",
        ],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    reference.to_csv(out_dir / "reference_features.csv", index=False)
    flat = {f"test_{k}": v for k, v in test_report.items() if isinstance(v, float)}
    flat.update({"test_ece": cal_report["ece"], "test_brier": cal_report["brier"]})
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:  # consumed by DVC
        json.dump(flat, f, indent=2)
    if args.metrics_out:  # separate file so DVC can track metrics outside the model directory
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(flat, indent=2), encoding="utf-8")
    Path(args.output_dir, "ACTIVE").write_text(version, encoding="utf-8")

    if not args.no_mlflow:
        log_to_mlflow(args, cfg, metadata, comparison, out_dir)
    return metadata


def log_to_mlflow(args, cfg, metadata, comparison, out_dir: Path) -> None:
    import os

    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=metadata["model_version"]):
        mlflow.log_params({
            "selected_model": metadata["selected_model"], "seed": metadata["seed"],
            "dataset_type": args.dataset, "synthetic_data": metadata["synthetic_data"],
            **{f"split_{k}": v for k, v in metadata["dataset"]["split_sizes"].items()},
            "word_ngrams": str(cfg["features"]["word_ngram_range"]),
            "char_ngrams": str(cfg["features"]["char_ngram_range"]),
            "calibration_method": cfg["calibration"]["method"],
        })
        for name, rep in comparison.items():
            for k in ("accuracy", "f1_macro", "roc_auc"):
                if k in rep:
                    mlflow.log_metric(f"val_{name}_{k}", rep[k])
        for k, v in metadata["test_metrics"].items():
            if isinstance(v, float):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("test_ece", metadata["calibration"]["ece"])
        mlflow.log_metric("test_brier", metadata["calibration"]["brier"])
        mlflow.log_metric("confidence_threshold", metadata["confidence_threshold"])
        mlflow.log_metric("novelty_threshold", metadata["novelty"]["threshold"])
        mlflow.set_tag("model_version", metadata["model_version"])
        mlflow.log_artifacts(str(out_dir), artifact_path="model_bundle")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the VeriNews classifier")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--dataset", choices=["isot", "csv"], default="isot")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--csv-path")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--version")
    p.add_argument("--seed", type=int)
    p.add_argument("--experiment", default="verinews-fake-news")
    p.add_argument("--metrics-out", help="also write flat test metrics here (used by DVC)")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="mark metadata as synthetic data")
    return p.parse_args(argv)


if __name__ == "__main__":
    meta = run(parse_args())
    summary = {
        "model_version": meta["model_version"],
        "selected_model": meta["selected_model"],
        "synthetic_data": meta["synthetic_data"],
        "test_metrics": {
            k: meta["test_metrics"].get(k) for k in ("accuracy", "f1_fake", "roc_auc")
        },
        "calibration": meta["calibration"],
    }
    print(json.dumps(summary, indent=2))
    sys.exit(0)
