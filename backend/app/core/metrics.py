"""In-process operational metrics. These describe the SERVICE (traffic, latency, errors),
never model accuracy: live accuracy needs reviewed ground-truth labels (see monitoring docs)."""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np


class MetricsCollector:
    def __init__(self, max_samples: int = 2000):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self._requests: Counter = Counter()
        self._errors: Counter = Counter()
        self._latency: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._predictions: Counter = Counter()
        self._verification: Counter = Counter()
        self._provider_failures: Counter = Counter()
        self.model_version: str | None = None

    def record_request(self, route: str, method: str, status: int, seconds: float) -> None:
        key = f"{method} {route}"
        with self._lock:
            self._requests[key] += 1
            if status >= 500:
                self._errors[f"{key}|5xx"] += 1
            elif status >= 400:
                self._errors[f"{key}|4xx"] += 1
            self._latency[key].append(seconds * 1000.0)

    def record_prediction(self, pred) -> None:
        with self._lock:
            self.model_version = pred.model_version
            self._predictions[f"label_{pred.label}"] += 1
            self._predictions["total"] += 1
            if pred.out_of_distribution:
                self._predictions["out_of_distribution"] += 1
            if pred.low_confidence:
                self._predictions["low_confidence"] += 1
            if pred.needs_verification:
                self._predictions["needs_verification"] += 1

    def record_verification(self, result: dict) -> None:
        with self._lock:
            self._verification["triggered"] += 1
            self._verification["completed" if result["completed"] else "incomplete"] += 1
            self._verification[f"verdict_{result['verdict']}"] += 1
            for p in result["providers"]:
                if p["status"] in ("error", "partial"):
                    self._provider_failures[p["name"]] += 1

    def snapshot(self) -> dict:
        with self._lock:
            routes = {}
            for key, count in self._requests.items():
                lat = list(self._latency[key])
                routes[key] = {
                    "count": count,
                    "errors_4xx": self._errors.get(f"{key}|4xx", 0),
                    "errors_5xx": self._errors.get(f"{key}|5xx", 0),
                    "latency_ms_p50": round(float(np.percentile(lat, 50)), 2) if lat else None,
                    "latency_ms_p95": round(float(np.percentile(lat, 95)), 2) if lat else None,
                }
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_requests": sum(self._requests.values()),
                "routes": routes,
                "model_version": self.model_version,
                "predictions": dict(self._predictions),
                "verification": dict(self._verification),
                "verification_provider_failures": dict(self._provider_failures),
                "note": "Operational metrics only. They do not measure model accuracy.",
            }
