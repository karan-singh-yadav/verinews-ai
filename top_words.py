import joblib
import numpy as np
from pathlib import Path

root = Path("models")
version = (root / "ACTIVE").read_text().strip()
clf = joblib.load(root / version / "classifier.joblib")
pipe = clf.calibrated_classifiers_[0].estimator.estimator
names = pipe.named_steps["features"].get_feature_names_out()
coef = pipe.named_steps["clf"].coef_.ravel()
words = [(n.replace("word__", ""), c) for n, c in zip(names, coef) if n.startswith("word__")]
words.sort(key=lambda x: x[1])
print("Model version:", version)
print("\nTop words pushing toward REAL:\n ", ", ".join(w for w, _ in words[:30]))
print("\nTop words pushing toward FAKE:\n ", ", ".join(w for w, _ in words[-30:][::-1]))