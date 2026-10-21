"""Optional user feedback, stored as JSON Lines.

Privacy: by default only a SHA-256 hash of the text is stored (enough to de-duplicate), never the
text. Text is stored only when the user explicitly opts in. Feedback is user-reported and
unverified: it is a candidate pool for expert review, NOT ground truth."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, *, user_label: str, predicted_label: str | None, model_version: str | None,
            request_id: str | None, comment: str | None, text: str | None,
            consent_store_text: bool) -> str:
        fid = uuid.uuid4().hex
        record = {
            "id": fid, "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "request_id": request_id, "model_version": model_version,
            "predicted_label": predicted_label, "user_label": user_label,
            "comment": (comment or "").strip()[:500] or None,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
            "text": text if (consent_store_text and text) else None,
            "reviewed": False, "reviewed_label": None, "reviewer": None, "reviewed_at": None,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return fid

    def summary(self) -> dict:
        counts = {"fake": 0, "real": 0, "unsure": 0}
        agree = disagree = 0
        total = 0
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    total += 1
                    counts[r["user_label"]] = counts.get(r["user_label"], 0) + 1
                    if r.get("predicted_label") and r["user_label"] in ("fake", "real"):
                        agree += r["predicted_label"] == r["user_label"]
                        disagree += r["predicted_label"] != r["user_label"]
        return {
            "total": total, "by_user_label": counts,
            "user_agreement_with_model": {"agree": agree, "disagree": disagree},
            "note": "User-reported and unverified. This is NOT live model accuracy; accuracy "
                    "requires expert-reviewed ground-truth labels.",
        }
