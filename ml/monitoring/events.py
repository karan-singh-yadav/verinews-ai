"""Prediction event log used for drift monitoring and analytics.

Privacy: events contain only derived numbers/flags (probabilities, word count, novelty score,
verdict). NEVER article text, URLs or IP addresses."""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["ts", "model_version", "label", "fake_probability", "confidence", "novelty_score",
           "word_count", "char_length", "ood", "low_confidence", "needs_verification",
           "verification_triggered", "verdict", "verification_completed"]


class PredictionLog:
    def __init__(self, path: str | Path, enabled: bool = True):
        self.path, self.enabled = Path(path), enabled
        self._lock = threading.Lock()

    def record(self, pred, verification: dict | None = None) -> None:
        if not self.enabled:
            return
        result = (verification or {}).get("result")
        event = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_version": pred.model_version, "label": pred.label,
            "fake_probability": round(pred.fake_probability, 6),
            "confidence": round(pred.confidence, 6), "novelty_score": round(pred.novelty_score, 6),
            "word_count": pred.word_count, "char_length": pred.char_length,
            "ood": pred.out_of_distribution, "low_confidence": pred.low_confidence,
            "needs_verification": pred.needs_verification,
            "verification_triggered": bool(verification and verification.get("triggered")),
            "verdict": result["verdict"] if result else None,
            "verification_completed": result["completed"] if result else None,
        }
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
        except OSError:  # monitoring must never break a user request
            pass

    def load(self, last_n: int = 2000, model_version: str | None = None) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        rows: deque = deque(maxlen=last_n)
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if model_version is None or r.get("model_version") == model_version:
                    rows.append(r)
        return pd.DataFrame(list(rows), columns=COLUMNS)
