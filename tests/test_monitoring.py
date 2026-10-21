import json

import numpy as np
import pandas as pd
import pytest

from ml import registry
from ml.inference.predictor import Predictor
from ml.monitoring import review
from ml.monitoring.drift import build_report, proportion_shift, psi
from ml.monitoring.events import COLUMNS, PredictionLog
from ml.monitoring.performance import evaluate_reviewed, wilson_interval

RNG = np.random.default_rng(0)


def frame(n, wc=(300, 80), nov=(0.3, 0.05), p_fake=(0.5, 0.3), seed=0):
    r = np.random.default_rng(seed)
    p = np.clip(r.normal(p_fake[0], p_fake[1], n), 0.001, 0.999)
    return pd.DataFrame({
        "word_count": np.clip(r.normal(*wc, n), 20, None).astype(int),
        "char_length": np.clip(r.normal(*wc, n), 20, None).astype(int) * 6,
        "fake_probability": p, "confidence": np.maximum(p, 1 - p),
        "novelty_score": np.clip(r.normal(*nov, n), 0, 1)})


def live(df, ood=0.05, low=0.1, fake=0.5, seed=1):
    r = np.random.default_rng(seed)
    out = df.copy()
    out["label"] = np.where(r.uniform(size=len(df)) < fake, "fake", "real")
    out["ood"] = r.uniform(size=len(df)) < ood
    out["low_confidence"] = r.uniform(size=len(df)) < low
    return out


RATES = {"n": 3000, "fake_rate": 0.5, "ood_rate": 0.05, "low_confidence_rate": 0.1}


# ---------------- statistics ----------------
def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    a, b = RNG.normal(0, 1, 5000), RNG.normal(0, 1, 5000)
    assert psi(a, b) < 0.05
    assert psi(a, RNG.normal(2, 1, 5000)) > 0.25
    assert psi(np.ones(100), np.ones(100)) < 0.01      # constant reference does not crash


def test_proportion_shift():
    diff, p = proportion_shift(0.05, 3000, 0.30, 500)
    assert diff == pytest.approx(0.25) and p < 1e-6
    assert proportion_shift(0.05, 3000, 0.05, 500)[1] > 0.9


def test_wilson_interval_contains_estimate_and_widens_with_small_n():
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi
    assert (hi - lo) > (wilson_interval(800, 1000)[1] - wilson_interval(800, 1000)[0])
    assert wilson_interval(0, 0) == (0.0, 0.0)


# ---------------- drift report ----------------
def test_no_drift_on_same_distribution():
    ref = frame(3000, seed=1)
    rep = build_report(ref, RATES, live(frame(600, seed=2)))
    assert rep["status"] == "ok", rep["message"]


def test_drift_detected_on_shifted_inputs_and_rates():
    ref = frame(3000, seed=1)
    shifted = live(frame(600, wc=(900, 150), nov=(0.6, 0.08), seed=3), ood=0.4)
    rep = build_report(ref, RATES, shifted)
    assert rep["status"] == "drift"
    flagged = {f["feature"] for f in rep["features"] if f["status"] == "drift"}
    assert {"word_count", "novelty_score"} <= flagged
    assert any(r["rate"] == "ood_rate" and r["status"] == "drift" for r in rep["rates"])


def test_insufficient_data_makes_no_drift_claim():
    rep = build_report(frame(3000), RATES, live(frame(20, wc=(2000, 10))))
    assert rep["status"] == "insufficient_data" and rep["features"] == []
    assert "No drift claim" in rep["message"]


def test_missing_reference_is_reported_not_crashed():
    rep = build_report(None, None, live(frame(500)))
    assert rep["status"] == "no_reference"


def test_report_always_states_drift_is_not_accuracy():
    assert "NOT a measurement of accuracy" in build_report(None, None, live(frame(5)))["interpretation"]


# ---------------- event log privacy ----------------
class P:
    model_version, label, fake_probability, confidence, novelty_score = "v1", "fake", 0.9, 0.9, 0.4
    word_count, char_length, out_of_distribution, low_confidence = 120, 700, True, False
    needs_verification = True


def test_event_log_stores_no_text_and_loads_back(tmp_path):
    log = PredictionLog(tmp_path / "e.jsonl")
    log.record(P(), {"triggered": True, "result": {"verdict": "Inconclusive", "completed": False}})
    log.record(P())
    raw = (tmp_path / "e.jsonl").read_text()
    assert set(json.loads(raw.splitlines()[0])) == set(COLUMNS)
    assert "text" not in raw and "url" not in raw
    df = log.load()
    assert len(df) == 2 and df.loc[0, "verdict"] == "Inconclusive"
    assert len(log.load(last_n=1)) == 1 and log.load(model_version="other").empty


def test_event_log_disabled_and_unwritable_do_not_break(tmp_path):
    PredictionLog(tmp_path / "x.jsonl", enabled=False).record(P())
    assert not (tmp_path / "x.jsonl").exists()
    blocked = tmp_path / "file"
    blocked.write_text("x")
    PredictionLog(blocked / "sub" / "e.jsonl").record(P())      # parent is a file -> OSError, swallowed


# ---------------- performance from reviewed labels ----------------
def write_feedback(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def rec(pred, rev, reviewed=True, version="v1"):
    return {"id": f"{pred}{rev}{np.random.randint(1e9)}", "predicted_label": pred,
            "user_label": "fake", "reviewed": reviewed, "reviewed_label": rev,
            "model_version": version}


def test_performance_refuses_to_report_without_enough_reviewed_labels(tmp_path):
    f = tmp_path / "fb.jsonl"
    write_feedback(f, [rec("fake", "fake") for _ in range(10)])
    out = evaluate_reviewed(f, min_labels=50)
    assert out["status"] == "insufficient_labels" and out["metrics"] is None
    assert "No accuracy claim" in out["message"]
    assert evaluate_reviewed(tmp_path / "missing.jsonl")["status"] == "insufficient_labels"


def test_performance_ignores_unreviewed_and_unsure_records(tmp_path):
    f = tmp_path / "fb.jsonl"
    rows = [rec("fake", "fake") for _ in range(60)]
    rows += [rec("fake", "real", reviewed=False) for _ in range(500)]      # user votes: ignored
    rows += [rec("fake", "unsure") for _ in range(40)]
    write_feedback(f, rows)
    out = evaluate_reviewed(f, min_labels=50)
    assert out["status"] == "ok" and out["reviewed_usable"] == 60
    assert out["metrics"]["accuracy"] == 1.0


def test_performance_metrics_match_hand_computation(tmp_path):
    f = tmp_path / "fb.jsonl"
    rows = ([rec("fake", "fake")] * 30 + [rec("fake", "real")] * 10     # tp=30 fp=10
            + [rec("real", "real")] * 50 + [rec("real", "fake")] * 10)  # tn=50 fn=10
    write_feedback(f, rows)
    m = evaluate_reviewed(f, min_labels=50)["metrics"]
    assert m["n"] == 100 and m["accuracy"] == pytest.approx(0.80)
    assert m["precision_fake"] == pytest.approx(0.75) and m["recall_fake"] == pytest.approx(0.75)
    assert m["confusion_matrix"] == [[50, 10], [10, 30]]
    assert m["accuracy_ci95"][0] < 0.80 < m["accuracy_ci95"][1]


def test_performance_filters_by_model_version(tmp_path):
    f = tmp_path / "fb.jsonl"
    write_feedback(f, [rec("fake", "fake", version="v1")] * 60 + [rec("fake", "real", version="v2")] * 60)
    assert evaluate_reviewed(f, 50, "v1")["metrics"]["accuracy"] == 1.0
    assert evaluate_reviewed(f, 50, "v2")["metrics"]["accuracy"] == 0.0


# ---------------- review CLI helpers ----------------
def test_review_workflow(tmp_path):
    f = tmp_path / "fb.jsonl"
    write_feedback(f, [
        {"id": "a1", "predicted_label": "fake", "user_label": "real", "text": "article body",
         "reviewed": False},
        {"id": "b2", "predicted_label": "real", "user_label": "fake", "text": None,
         "reviewed": False}])
    assert [r["id"] for r in review.list_unreviewed(f)] == ["a1"]            # hash-only skipped
    assert len(review.list_unreviewed(f, include_no_text=True)) == 2
    assert review.set_review(f, "a1", "fake", "alice") is True
    done = json.loads(f.read_text().splitlines()[0])
    assert done["reviewed"] and done["reviewed_label"] == "fake" and done["reviewer"] == "alice"
    assert review.set_review(f, "zzz", "fake", "alice") is False
    with pytest.raises(ValueError):
        review.set_review(f, "a1", "maybe", "alice")


# ---------------- registry ----------------
def test_registry_lists_promotes_and_rolls_back(trained_models, tmp_path):
    models_dir, _ = trained_models
    import shutil
    root = tmp_path / "models"
    shutil.copytree(models_dir, root)
    shutil.copytree(root / "vtest", root / "vold")
    meta = json.loads((root / "vold" / "metadata.json").read_text())
    meta["model_version"], meta["trained_at"] = "vold", "2020-01-01T00:00:00+00:00"
    (root / "vold" / "metadata.json").write_text(json.dumps(meta))
    versions = registry.list_versions(root)
    assert [v["model_version"] for v in versions] == ["vtest", "vold"]       # newest first
    assert versions[0]["active"] and versions[0]["synthetic_data"] is True
    registry.promote(root, "vold")
    assert registry.active_version(root) == "vold"
    assert Predictor.load(root).version == "vold"
    for bad in ("../etc", "nope", ".hidden"):
        with pytest.raises((ValueError, FileNotFoundError)):
            registry.promote(root, bad)


def test_training_writes_reference_profile_and_dvc_metrics(trained_models):
    models_dir, meta = trained_models
    ref = pd.read_csv(models_dir / "vtest" / "reference_features.csv")
    assert list(ref.columns) == ["word_count", "char_length", "fake_probability",
                                 "confidence", "novelty_score"]
    assert len(ref) > 0 and not ref.isna().any().any()
    mr = meta["monitoring_reference"]
    assert 0 <= mr["ood_rate"] <= 1 and mr["n"] == meta["dataset"]["split_sizes"]["test"]
    metrics = json.loads((models_dir / "vtest" / "metrics.json").read_text())
    assert metrics["test_accuracy"] == pytest.approx(meta["test_metrics"]["accuracy"])
    assert Predictor.load(models_dir).reference_features() is not None


def test_prediction_exposes_text_stats(trained_models):
    p = Predictor.load(trained_models[0]).predict(
        "the finance ministry confirmed that quarterly revenue rose by 5 percent compared "
        "with last year and officials said the committee released a detailed report")
    assert p.word_count >= 20 and p.char_length > p.word_count


# ---------------- regression: small-reference false alarms (measured 97% before the fix) ----
def test_small_reference_does_not_cry_wolf_on_same_distribution():
    bad = sum(
        build_report(frame(90, seed=s * 2), {**RATES, "n": 90},
                     live(frame(300, seed=s * 2 + 1), seed=s))["status"] != "ok"
        for s in range(60))
    assert bad / 60 <= 0.25      # measured ~8%; loose bound so the test is not flaky


def test_small_reference_still_detects_real_shift_and_says_it_is_noisy():
    rep = build_report(frame(90, seed=1), {**RATES, "n": 90},
                       live(frame(300, nov=(0.5, 0.05), seed=2), ood=0.4, seed=3))
    assert rep["status"] == "drift" and rep["reference_small"] is True
    assert "noisy" in rep["message"]


def test_adaptive_bins_scale_with_sample_size():
    from ml.monitoring.drift import adaptive_bins
    assert adaptive_bins(90, 300) == 3 and adaptive_bins(5000, 5000) == 10
    assert adaptive_bins(5000, 120) == 3
