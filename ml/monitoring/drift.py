"""Data / prediction drift statistics (reference = test split at training time, current = live
traffic). Drift means "inputs or outputs look different from training time". It does NOT mean
accuracy dropped; measuring accuracy needs reviewed ground-truth labels (see performance.py).

Per-feature: PSI (effect size; <0.1 stable, 0.1-0.25 moderate, >=0.25 large) and a two-sample
KS test (significance). With big samples KS flags trivial shifts, so PSI decides the status.
Rates (fake / out-of-distribution / low-confidence share): two-proportion z-test plus a minimum
practical difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ["word_count", "char_length", "fake_probability", "confidence", "novelty_score"]


@dataclass
class DriftConfig:
    min_samples: int = 100
    psi_warn: float = 0.10
    psi_drift: float = 0.25
    ks_alpha: float = 0.01        # needed (with PSI >= psi_drift) to call "drift"
    ks_warn_alpha: float = 0.05   # needed (with PSI >= psi_warn) to call "warning"
    small_reference: int = 300    # below this the report carries a "noisy statistics" note
    rate_alpha: float = 0.01
    min_rate_diff: float = 0.05


def adaptive_bins(n_ref: int, n_cur: int) -> int:
    """~50+ observations per bin on the smaller sample; PSI is badly biased upward otherwise."""
    return int(max(3, min(10, min(n_ref, n_cur) // 50)))


def psi(reference, current, bins: int | None = None, eps: float = 1e-4) -> float:
    """Population Stability Index with reference-quantile bins."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    bins = bins or adaptive_bins(len(ref), len(cur))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant reference: compare "same value" share instead
        edges = np.array([-np.inf, np.median(ref), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(ref, edges)[0] / len(ref)
    cur_p = np.histogram(cur, edges)[0] / len(cur)
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def proportion_shift(
    ref_rate: float, n_ref: int, cur_rate: float, n_cur: int
) -> tuple[float, float]:
    """Two-proportion z-test. Returns (difference, p_value)."""
    pooled = (ref_rate * n_ref + cur_rate * n_cur) / (n_ref + n_cur)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_cur))
    if se == 0:
        return cur_rate - ref_rate, 1.0
    z = (cur_rate - ref_rate) / se
    return float(cur_rate - ref_rate), float(2 * (1 - stats.norm.cdf(abs(z))))


def _feature_status(p: float, ks_p: float, cfg: DriftConfig) -> str:
    if p >= cfg.psi_drift and ks_p < cfg.ks_alpha:
        return "drift"
    if p >= cfg.psi_warn and ks_p < cfg.ks_warn_alpha:
        return "warning"
    return "ok"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, cfg: DriftConfig) -> list[dict]:
    rows = []
    for f in FEATURES:
        if f not in ref or f not in cur:
            continue
        r, c = ref[f].dropna(), cur[f].dropna()
        if len(r) < 2 or len(c) < 2:
            continue
        p = psi(r, c)
        ks = stats.ks_2samp(r, c)
        rows.append({
            "feature": f, "psi": round(p, 4), "ks_statistic": round(float(ks.statistic), 4),
            "ks_p_value": float(ks.pvalue), "status": _feature_status(p, float(ks.pvalue), cfg),
            "reference_mean": round(float(r.mean()), 4), "current_mean": round(float(c.mean()), 4),
            "reference_median": round(float(r.median()), 4),
            "current_median": round(float(c.median()), 4),
        })
    return rows


def rate_drift(ref_rates: dict, n_ref: int, cur: pd.DataFrame, cfg: DriftConfig) -> list[dict]:
    current = {
        "fake_rate": float((cur["label"] == "fake").mean()),
        "ood_rate": float(cur["ood"].astype(bool).mean()),
        "low_confidence_rate": float(cur["low_confidence"].astype(bool).mean()),
    }
    rows = []
    for name, cur_rate in current.items():
        if name not in ref_rates:
            continue
        diff, pval = proportion_shift(ref_rates[name], n_ref, cur_rate, len(cur))
        drifted = pval < cfg.rate_alpha and abs(diff) >= cfg.min_rate_diff
        rows.append({"rate": name, "reference": round(ref_rates[name], 4),
                     "current": round(cur_rate, 4), "difference": round(diff, 4),
                     "p_value": pval, "status": "drift" if drifted else "ok"})
    return rows


INTERPRETATION = (
    "Drift compares live inputs/outputs with the training-time reference. It signals that the "
    "model may be seeing different data; it is NOT a measurement of accuracy. Use reviewed "
    "labels (/api/monitoring/performance) to measure real-world performance."
)


def build_report(reference: pd.DataFrame | None, ref_rates: dict | None, current: pd.DataFrame,
                 cfg: DriftConfig | None = None) -> dict:
    cfg = cfg or DriftConfig()
    base = {"n_current": int(len(current)), "min_samples": cfg.min_samples,
            "interpretation": INTERPRETATION}
    if reference is None:
        return {**base, "status": "no_reference", "n_reference": 0, "features": [], "rates": [],
                "message": "This model version has no saved reference profile. "
                           "Retrain to create one."}
    base["n_reference"] = int((ref_rates or {}).get("n", len(reference)))
    if len(current) < cfg.min_samples:
        return {**base, "status": "insufficient_data", "features": [], "rates": [],
                "message": f"Need at least {cfg.min_samples} logged requests to assess drift "
                           f"(have {len(current)}). No drift claim is made."}
    features = feature_drift(reference, current, cfg)
    rates = rate_drift(ref_rates or {}, base["n_reference"], current, cfg) if ref_rates else []
    statuses = [x["status"] for x in features + rates]
    status = "drift" if "drift" in statuses else "warning" if "warning" in statuses else "ok"
    flagged = [x.get("feature") or x.get("rate") for x in features + rates
               if x["status"] in ("drift", "warning")]
    msg = ("No significant drift detected." if status == "ok"
           else "Shift detected in: " + ", ".join(flagged) + ". Review recent inputs.")
    out = {**base, "status": status, "features": features, "rates": rates, "message": msg}
    if base["n_reference"] < cfg.small_reference:
        out["reference_small"] = True
        out["message"] += (f" Note: the reference profile has only {base['n_reference']} rows, "
                           "so statistics are noisy; train on more data for a stable baseline.")
    return out
