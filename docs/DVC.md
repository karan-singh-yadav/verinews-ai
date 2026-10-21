# Reproducible training with DVC

`dvc.yaml` defines one stage (`train`) with the dataset files, `ml/` code and
`configs/training.yaml` as dependencies; outputs are `models/current` + `models/ACTIVE`;
`metrics/train.json` is the tracked metrics file.

```powershell
pip install -r requirements-monitoring.txt
git init                      # DVC needs a git repo (skip if already one)
dvc init
# put Fake.csv and True.csv in data\raw\ first
dvc repro                     # trains; re-running with no changes is skipped
dvc metrics show              # test metrics of the last run
dvc params diff               # config changes vs last commit
git add dvc.yaml dvc.lock metrics .dvc .dvcignore models/.gitignore
```
Verified in this project's sandbox on synthetic data: `dvc dag`, `dvc repro`, skip-when-unchanged,
re-run after a config change, and `dvc metrics show`. **Not verified:** remote storage
(`dvc remote add` / `dvc push`) and tracking `data/raw` with `dvc add`.
MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) keeps full run history; DVC
reproduces the pipeline.
