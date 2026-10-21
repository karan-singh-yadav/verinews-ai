"""Offline drift report:  python -m ml.monitoring.report [--evidently]
Writes reports/drift.json (always) and reports/drift_evidently.html (with --evidently)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.inference.predictor import Predictor
from ml.monitoring.drift import DriftConfig, build_report
from ml.monitoring.events import PredictionLog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--events", default="data/monitoring/events.jsonl")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--evidently", action="store_true")
    a = ap.parse_args()

    predictor = Predictor.load(a.models)
    current = PredictionLog(a.events).load(a.window, predictor.version)
    ref = predictor.reference_features()
    rates = predictor.metadata.get("monitoring_reference")
    report = build_report(ref, rates, current, DriftConfig(min_samples=a.min_samples))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{report['status']}] {report['message']}  -> {out / 'drift.json'}")
    if a.evidently:
        if ref is None or current.empty:
            print("Skipping Evidently: no reference or no current events.")
        else:
            from ml.monitoring.evidently_report import write_html_report
            print("Wrote", write_html_report(ref, current, out / "drift_evidently.html"))


if __name__ == "__main__":
    main()
