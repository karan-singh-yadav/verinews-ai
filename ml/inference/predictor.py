"""Inference wrapper: classification + uncertainty + novelty + 'should we verify online?'.

The ML output is a style/vocabulary-based estimate. It is NOT evidence that a claim is true.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib

from ml.data import LABEL_NAMES
from ml.preprocessing import clean_text, word_count

MIN_WORDS_FOR_RELIABLE_PREDICTION = 20


class InvalidInputError(ValueError):
    """Raised for empty / unusable input text."""


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model artifacts can be found."""


@dataclass
class Prediction:
    label: str
    fake_probability: float
    confidence: float
    confidence_validated: bool
    low_confidence: bool
    novelty_score: float
    novelty_threshold: float
    out_of_distribution: bool
    needs_verification: bool
    verification_reasons: list[str] = field(default_factory=list)
    model_version: str = ""
    word_count: int = 0
    char_length: int = 0
    confidence_threshold: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def __init__(self, model_dir: Path, classifier, novelty, metadata: dict):
        self.model_dir = model_dir
        self.classifier = classifier
        self.novelty = novelty
        self.metadata = metadata
        self.version = metadata["model_version"]
        self.confidence_threshold = float(metadata["confidence_threshold"])
        self.confidence_validated = bool(metadata["calibration"]["valid"])
        self._remove_leaks = bool(metadata["config"]["data"]["remove_source_leaks"])

    @classmethod
    def load(cls, models_root: str | Path = "models", version: str | None = None) -> Predictor:
        root = Path(models_root)
        if version is None:
            active = root / "ACTIVE"
            if not active.exists():
                raise ModelNotAvailableError(
                    f"No trained model found in '{root}'. Run training first (see README)."
                )
            version = active.read_text(encoding="utf-8").strip()
        model_dir = root / version
        try:
            metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            classifier = joblib.load(model_dir / "classifier.joblib")
            novelty = joblib.load(model_dir / "novelty.joblib")
        except (FileNotFoundError, OSError) as exc:
            raise ModelNotAvailableError(f"Model artifacts missing in '{model_dir}'") from exc
        return cls(model_dir, classifier, novelty, metadata)

    def reference_features(self):
        """Reference feature table saved at training time (None for models trained before
        monitoring existed)."""
        import pandas as pd

        path = self.model_dir / "reference_features.csv"
        return pd.read_csv(path) if path.exists() else None

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            raise InvalidInputError("No text provided")
        cleaned = [clean_text(t, self._remove_leaks) for t in texts]
        for raw, c in zip(texts, cleaned, strict=True):
            if not isinstance(raw, str) or not c:
                raise InvalidInputError("Text is empty after cleaning")

        p_fake = self.classifier.predict_proba(cleaned)[:, 1]
        nov_scores = self.novelty.score(cleaned)
        ood_flags = self.novelty.is_ood(nov_scores)

        results = []
        for text, p, ns, ood in zip(cleaned, p_fake, nov_scores, ood_flags, strict=True):
            p = float(p)
            label_id = int(p >= 0.5)
            confidence = p if label_id == 1 else 1.0 - p
            low_conf = confidence < self.confidence_threshold
            too_short = word_count(text) < MIN_WORDS_FOR_RELIABLE_PREDICTION

            reasons = []
            if low_conf:
                reasons.append(
                    f"Model confidence {confidence:.2f} is below the validated threshold "
                    f"{self.confidence_threshold:.2f}."
                )
            if ood:
                reasons.append(
                    f"Text looks unfamiliar compared with the training data "
                    f"(novelty {ns:.3f} > threshold {self.novelty.threshold:.3f})."
                )
            if too_short:
                reasons.append(
                    f"Text has fewer than {MIN_WORDS_FOR_RELIABLE_PREDICTION} words; "
                    "the classifier is unreliable on very short input."
                )
            results.append(
                Prediction(
                    label=LABEL_NAMES[label_id],
                    fake_probability=p,
                    confidence=float(confidence),
                    confidence_validated=self.confidence_validated,
                    low_confidence=bool(low_conf),
                    novelty_score=float(ns),
                    novelty_threshold=float(self.novelty.threshold),
                    out_of_distribution=bool(ood),
                    needs_verification=bool(reasons),
                    verification_reasons=reasons,
                    model_version=self.version,
                    word_count=word_count(text),
                    char_length=len(text),
                    confidence_threshold=self.confidence_threshold,
                )
            )
        return results
