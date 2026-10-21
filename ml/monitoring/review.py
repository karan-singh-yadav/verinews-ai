"""Expert review CLI: attach ground-truth labels to stored feedback.

  python -m ml.monitoring.review --list
  python -m ml.monitoring.review --label <id> fake --reviewer alice

Only records whose article text was stored with user consent can be meaningfully reviewed
(otherwise only a hash exists). Label by checking fact-checks/primary sources, not by trusting
the model or the submitting user."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def list_unreviewed(path: str | Path, limit: int = 20, include_no_text: bool = False) -> list[dict]:
    rows = [r for r in _read(Path(path)) if not r.get("reviewed")
            and (include_no_text or r.get("text"))]
    return rows[:limit]


def set_review(path: str | Path, feedback_id: str, label: str, reviewer: str) -> bool:
    if label not in ("fake", "real", "unsure"):
        raise ValueError("label must be fake, real or unsure")
    path = Path(path)
    rows, found = _read(path), False
    for r in rows:
        if r.get("id") == feedback_id:
            r.update(reviewed=True, reviewed_label=label, reviewer=reviewer,
                     reviewed_at=datetime.now(UTC).isoformat(timespec="seconds"))
            found = True
    if found:  # atomic rewrite
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        os.replace(tmp, path)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="Review stored feedback")
    ap.add_argument("--feedback", default="data/feedback/feedback.jsonl")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--include-no-text", action="store_true")
    ap.add_argument("--label", nargs=2, metavar=("ID", "LABEL"))
    ap.add_argument("--reviewer", default="")
    a = ap.parse_args()
    if a.list:
        for r in list_unreviewed(a.feedback, a.limit, a.include_no_text):
            print(f"[{r['id']}] model said: {r['predicted_label']} | user said: {r['user_label']}")
            print("   " + ((r.get("text") or "(no text stored)")[:300]).replace("\n", " "))
    elif a.label:
        if not a.reviewer:
            ap.error("--reviewer is required so labels are attributable")
        ok = set_review(a.feedback, a.label[0], a.label[1], a.reviewer)
        print("Updated." if ok else "No record with that id.")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
