import json

import numpy as np
import pytest

from ml.evaluation.metrics import expected_calibration_error
from ml.inference.predictor import InvalidInputError, ModelNotAvailableError, Predictor

REAL_LIKE = ("the finance ministry confirmed that quarterly revenue rose by 5 percent compared "
             "with last year and officials said the committee released a detailed report")
UNRELATED = ("simmer the onions with garlic then stir in tomato paste basil and oregano "
             "before folding the pasta into the warm skillet with olive oil and parmesan cheese")


def test_training_writes_artifacts_and_metadata(trained_models):
    models_dir, meta = trained_models
    assert (models_dir / "ACTIVE").read_text() == "vtest"
    for f in ("classifier.joblib", "novelty.joblib", "metadata.json"):
        assert (models_dir / "vtest" / f).exists()
    on_disk = json.loads((models_dir / "vtest" / "metadata.json").read_text())
    assert on_disk["synthetic_data"] is True
    assert set(on_disk["validation_comparison"]) >= {"logistic_regression", "linear_svm"}
    assert on_disk["dataset"]["split_sizes"]["test"] > 0


def test_metadata_metrics_are_internally_consistent(trained_models):
    _, meta = trained_models
    cm = np.array(meta["test_metrics"]["confusion_matrix"])
    assert cm.sum() == meta["test_metrics"]["n_samples"]
    acc = np.trace(cm) / cm.sum()
    assert acc == pytest.approx(meta["test_metrics"]["accuracy"])


def test_training_is_reproducible(trained_models, tmp_path):
    from ml.training.train import parse_args, run
    models_dir, meta = trained_models
    data_dir = models_dir.parent / "data"
    meta2 = run(parse_args(["--data-dir", str(data_dir), "--output-dir", str(tmp_path),
                            "--version", "again", "--no-mlflow", "--synthetic"]))
    assert meta2["test_metrics"]["confusion_matrix"] == meta["test_metrics"]["confusion_matrix"]
    assert meta2["novelty"]["threshold"] == pytest.approx(meta["novelty"]["threshold"])


def test_predictor_returns_valid_structure(trained_models):
    p = Predictor.load(trained_models[0])
    out = p.predict(REAL_LIKE)
    assert out.label in {"real", "fake"}
    assert 0.0 <= out.fake_probability <= 1.0
    assert 0.5 <= out.confidence <= 1.0
    assert out.model_version == "vtest"


def test_unfamiliar_text_is_flagged_ood_and_triggers_verification(trained_models):
    p = Predictor.load(trained_models[0])
    familiar, unfamiliar = p.predict(REAL_LIKE), p.predict(UNRELATED)
    assert unfamiliar.novelty_score > familiar.novelty_score
    assert unfamiliar.out_of_distribution is True
    assert unfamiliar.needs_verification is True
    assert any("unfamiliar" in r for r in unfamiliar.verification_reasons)


def test_ood_and_low_confidence_are_separate_indicators(trained_models):
    out = Predictor.load(trained_models[0]).predict(UNRELATED)
    assert isinstance(out.low_confidence, bool) and isinstance(out.out_of_distribution, bool)


def test_very_short_text_triggers_verification(trained_models):
    out = Predictor.load(trained_models[0]).predict("vaccine causes outbreak")
    assert out.needs_verification
    assert any("fewer than" in r for r in out.verification_reasons)


@pytest.mark.parametrize("bad", ["", "   ", "<b></b>", None])
def test_invalid_inputs_raise(trained_models, bad):
    with pytest.raises(InvalidInputError):
        Predictor.load(trained_models[0]).predict(bad)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        Predictor.load(tmp_path)


def test_ece_perfectly_calibrated_is_small_and_overconfident_is_large():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 20000)
    y_ok = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y_ok, p) < 0.02
    y_bad = (rng.uniform(size=p.size) < 0.5).astype(int)
    assert expected_calibration_error(y_bad, p) > 0.15
