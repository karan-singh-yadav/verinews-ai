"""Real-world performance from EXPERT-REVIEWED labels only.

Live accuracy is never inferred from confidence, user votes or verification verdicts. A feedback
record counts only after a reviewer assigns `reviewed_label` (see ml/monitoring/review.py).
Below `min_labels` reviewed examples no metric is reported at all.

Caveat that is always returned: feedback is self-selected, so even reviewed labels are not a
random sample of traffic. For an unbiased estimate, review a RANDOM sample of requests.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

NOTE = ("Computed only from records a reviewer labelled. Feedback is self-selected, so this is "
        "not guaranteed to represent all traffic; review a random sample for an unbiased "
        "estimate.")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _scores(rows: list[dict]) -> dict:
    tp = sum(r["predicted_label"] == "fake" and r["reviewed_label"] == "fake" for r in rows)
    fp = sum(r["predicted_label"] == "fake" and r["reviewed_label"] == "real" for r in rows)
    fn = sum(r["predicted_label"] == "real" and r["reviewed_label"] == "fake" for r in rows)
    tn = sum(r["predicted_label"] == "real" and r["reviewed_label"] == "real" for r in rows)
    n = len(rows)
    lo, hi = wilson_interval(tp + tn, n)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"n": n, "accuracy": (tp + tn) / n, "accuracy_ci95": [round(lo, 4), round(hi, 4)],
            "precision_fake": prec, "recall_fake": rec, "f1_fake": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]], "confusion_matrix_labels": ["real", "fake"]}


def evaluate_reviewed(feedback_path: str | Path, min_labels: int = 50,
                      model_version: str | None = None) -> dict:
    path = Path(feedback_path)
    records: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    usable = [r for r in records
              if r.get("reviewed") and r.get("reviewed_label") in ("fake", "real")
              and r.get("predicted_label") in ("fake", "real")
              and (model_version is None or r.get("model_version") == model_version)]
    out = {"total_feedback": len(records),
           "reviewed_usable": len(usable), "min_labels": min_labels, "note": NOTE}
    if len(usable) < min_labels:
        return {**out, "status": "insufficient_labels", "metrics": None,
                "message": f"Need at least {min_labels} expert-reviewed examples to report "
                           f"performance (have {len(usable)}). No accuracy claim is made."}
    by_version: dict[str, list[dict]] = {}
    for r in usable:
        by_version.setdefault(r.get("model_version") or "unknown", []).append(r)
    return {**out, "status": "ok", "metrics": _scores(usable),
            "by_model_version": {v: _scores(rs) for v, rs in by_version.items()
                                 if len(rs) >= min_labels}}
