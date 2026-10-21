"""Model versioning: list trained versions, promote one to ACTIVE, roll back.

Versions live in models/<version>/ with metadata.json. models/ACTIVE holds the active version id.
CLI:  python -m ml.registry list
      python -m ml.registry promote <version>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def active_version(models_root: str | Path) -> str | None:
    f = Path(models_root) / "ACTIVE"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


def list_versions(models_root: str | Path) -> list[dict]:
    """Summaries built from each version's recorded metadata (newest first)."""
    root, active, out = Path(models_root), active_version(models_root), []
    if not root.exists():
        return out
    for d in root.iterdir():
        meta_file = d / "metadata.json"
        if not (d.is_dir() and meta_file.exists()):
            continue
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tm = m.get("test_metrics", {})
        out.append({
            "model_version": m.get("model_version", d.name), "active": d.name == active,
            "selected_model": m.get("selected_model"), "trained_at": m.get("trained_at"),
            "synthetic_data": m.get("synthetic_data", False),
            "split_sizes": m.get("dataset", {}).get("split_sizes"),
            "test_metrics": {k: tm.get(k) for k in
                             ("accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc")},
            "calibration": m.get("calibration"),
            "validation_comparison": {
                n: {k: v.get(k) for k in ("accuracy", "f1_macro", "roc_auc")}
                for n, v in m.get("validation_comparison", {}).items()},
        })
    return sorted(out, key=lambda r: r["trained_at"] or "", reverse=True)


def promote(models_root: str | Path, version: str) -> None:
    root = Path(models_root)
    if "/" in version or "\\" in version or version.startswith("."):
        raise ValueError("Invalid version name")
    if not (root / version / "metadata.json").exists():
        raise FileNotFoundError(f"Model version '{version}' not found in {root}")
    (root / "ACTIVE").write_text(version, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="VeriNews model registry")
    ap.add_argument("command", choices=["list", "promote"])
    ap.add_argument("version", nargs="?")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    if a.command == "list":
        for v in list_versions(a.models):
            flag = "*" if v["active"] else " "
            tag = " [SYNTHETIC]" if v["synthetic_data"] else ""
            print(f"{flag} {v['model_version']}  {v['selected_model']}  {v['trained_at']}{tag}")
    else:
        if not a.version:
            ap.error("promote needs a version")
        promote(a.models, a.version)
        print(f"Active model is now {a.version}. Restart the API to load it.")


if __name__ == "__main__":
    main()
