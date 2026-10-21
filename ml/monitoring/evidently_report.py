"""Optional rich HTML drift report using Evidently (pip install -r requirements-monitoring.txt).
The API's own drift numbers (drift.py) do not depend on Evidently."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.monitoring.drift import FEATURES


def write_html_report(reference: pd.DataFrame, current: pd.DataFrame, out_path: str | Path) -> Path:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Evidently is not installed. Run: pip install evidently") from exc
    cols = [c for c in FEATURES if c in reference and c in current]
    snapshot = Report([DataDriftPreset()]).run(current[cols], reference[cols])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))
    return out
