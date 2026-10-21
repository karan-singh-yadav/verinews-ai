"""Create dashboard examples from YOUR dataset (held-out test articles only).

  python scripts/make_examples.py                      # uses data/raw and the active model
  python scripts/make_examples.py --per-class 10

Writes frontend/public/examples.json (git-ignored: it contains dataset text). Refresh the dashboard
afterwards. Nothing is written unless the held-out split is reproduced exactly (see ml/examples.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.examples import ExampleError, build_examples, reconstruct_splits, write_examples  # noqa: E402
from ml.inference.predictor import ModelNotAvailableError, Predictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="frontend/public/examples.json")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--wrong-per-class", type=int, default=1,
                    help="mix in this many articles the model gets wrong (shown unlabelled)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    try:
        predictor = Predictor.load(a.models)
        print(f"Active model: {predictor.version}. Rebuilding its held-out test split...")
        _, _, test, acc = reconstruct_splits(predictor, a.data_dir)
    except (ModelNotAvailableError, ExampleError, FileNotFoundError) as exc:
        print(f"\nCould not create examples: {exc}")
        return 1
    examples = build_examples(test, a.per_class, a.wrong_per_class, a.seed)
    out = write_examples(a.out, examples, predictor.version, acc)
    n_real = sum(e["label"] == "real" for e in examples)
    print(f"Verified: the rebuilt test split reproduces the recorded accuracy ({acc:.4%}).")
    print(f"Wrote {len(examples)} examples ({n_real} real, {len(examples) - n_real} fake) to {out}")
    print("Refresh the dashboard and use 'Try an article from your dataset' on the Check page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())