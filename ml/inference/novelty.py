"""Out-of-distribution (novelty) detection.

Method: embed documents with TF-IDF -> truncated SVD (LSA) -> L2 normalise, then score each
document by its mean cosine distance to its k nearest *training* documents. A high score means
"this text looks unlike anything the model was trained on".

The threshold is a percentile of the same score computed on held-out VALIDATION documents,
so it reflects the real spread of in-distribution data.

Limitations (documented in the UI too): LSA captures vocabulary/topic overlap, not deep
semantics. It flags unfamiliar topics and wording; it cannot tell whether a claim is true.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer


@dataclass
class NoveltyDetector:
    svd_components: int = 128
    max_features: int = 50000
    k_neighbors: int = 5
    max_reference_docs: int = 20000
    seed: int = 42
    threshold: float | None = None
    threshold_percentile: float | None = None
    _vec: TfidfVectorizer | None = field(default=None, repr=False)
    _svd: TruncatedSVD | None = field(default=None, repr=False)
    _norm: Normalizer | None = field(default=None, repr=False)
    _nn: NearestNeighbors | None = field(default=None, repr=False)

    def fit(self, texts: list[str]) -> NoveltyDetector:
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=self.max_features, sublinear_tf=True,
            min_df=2 if len(texts) > 50 else 1,
        )
        tfidf = self._vec.fit_transform(texts)
        n_comp = max(2, min(self.svd_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        self._norm = Normalizer(copy=False)
        emb = self._norm.fit_transform(self._svd.fit_transform(tfidf))
        if emb.shape[0] > self.max_reference_docs:
            idx = np.random.default_rng(self.seed).choice(
                emb.shape[0], self.max_reference_docs, replace=False
            )
            emb = emb[idx]
        self._nn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, emb.shape[0]), metric="cosine", algorithm="brute"
        ).fit(emb.astype(np.float32))
        return self

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("NoveltyDetector is not fitted")
        return self._norm.transform(self._svd.transform(self._vec.transform(texts)))

    def score(self, texts: list[str]) -> np.ndarray:
        """Mean cosine distance to the k nearest training docs (higher = more unfamiliar)."""
        emb = self._embed(texts).astype(np.float32)
        dist, _ = self._nn.kneighbors(emb)
        return dist.mean(axis=1)

    def calibrate_threshold(self, val_texts: list[str], percentile: float) -> float:
        scores = self.score(val_texts)
        self.threshold = float(np.percentile(scores, percentile))
        self.threshold_percentile = percentile
        return self.threshold

    def is_ood(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Threshold not calibrated; call calibrate_threshold() first")
        return scores > self.threshold
