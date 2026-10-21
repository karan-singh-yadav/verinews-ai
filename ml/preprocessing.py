"""Text cleaning shared by training and inference (identical code path = no train/serve skew)."""
from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Known dataset artifacts that let a model "cheat" instead of reading content.
# In the popular ISOT/Kaggle dataset, real articles are Reuters wire copy that start with
# "CITY (Reuters) -", so a classifier can score highly just by spotting that prefix.
_SOURCE_LEAK_PATTERNS = [
    re.compile(r"^\s*[A-Z][A-Za-z .,'/&-]{1,60}\(\s*Reuters\s*\)\s*[-\u2013\u2014]?\s*"),
    re.compile(r"\(\s*reuters\s*\)", re.I),
    re.compile(r"\breuters\b", re.I),
    re.compile(r"21st century wire says", re.I),
    re.compile(r"featured image (via|by)[^.\n]*", re.I),
    re.compile(r"\bimage via [^.\n]*", re.I),
    re.compile(r"pic\.twitter\.com/\S+", re.I),
    re.compile(r"\bgetty images?\b", re.I),
    re.compile(r"\bphoto by [^.\n]*", re.I),
]


def strip_source_artifacts(text: str) -> str:
    """Remove publisher-specific boilerplate that leaks the label in some datasets."""
    for pattern in _SOURCE_LEAK_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def clean_text(text: object, remove_source_leaks: bool = True) -> str:
    """Normalize raw article text. Never raises on None/NaN/non-str; returns '' instead."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML.sub(" ", text)
    if remove_source_leaks:
        text = strip_source_artifacts(text)
    text = _URL.sub(" url ", text)
    text = _EMAIL.sub(" email ", text)
    text = text.lower()
    text = _WS.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())
