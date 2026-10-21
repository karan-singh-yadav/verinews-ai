import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    """Train once on the SYNTHETIC fixture. Used to test plumbing, never to judge accuracy."""
    base = tmp_path_factory.mktemp("artifacts")
    data_dir, models_dir = base / "data", base / "models"
    subprocess.run(
        [sys.executable, "scripts/make_synthetic_fixture.py", "--out", str(data_dir),
         "--n-per-class", "300"], cwd=ROOT, check=True)
    from ml.training.train import parse_args, run
    meta = run(parse_args([
        "--config", str(ROOT / "configs/training.yaml"), "--dataset", "isot",
        "--data-dir", str(data_dir), "--output-dir", str(models_dir),
        "--version", "vtest", "--no-mlflow", "--synthetic"]))
    return models_dir, meta
