from __future__ import annotations

import abc
import html
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

from backend.app.services.claims import Claim

_TAG = re.compile(r"<[^>]+>")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WS = re.compile(r"\s+")
_TWO_LEVEL = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.in", "org.in", "gov.in",
              "nic.in", "co.za", "com.br", "co.jp", "co.nz"}


class ProviderError(Exception):
    """Base class; messages are safe to show to users (never contain keys or URLs with keys)."""


class ProviderNotConfigured(ProviderError):
    pass


class ProviderUnavailable(ProviderError):
    pass


def sanitize_text(value: object, max_len: int = 500) -> str:
    """Retrieved content is untrusted: strip markup/control chars and bound the length."""
    if value is None:
        return ""
    s = html.unescape(str(value))
    s = _TAG.sub(" ", s)
    s = _CTRL.sub("", s)
    return _WS.sub(" ", s).strip()[:max_len]


def safe_url(value: object) -> str | None:
    """Only http(s) links are ever surfaced (blocks javascript:, data:, etc.)."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        p = urlsplit(value.strip())
    except ValueError:
        return None
    return value.strip() if p.scheme in ("http", "https") and p.hostname else None


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def registered_domain(domain: str) -> str:
    domain = domain.lower().strip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    labels = domain.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else domain


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class SearchResult:
    provider: str
    kind: str                 # "fact_check" | "news"
    title: str
    url: str
    snippet: str
    publisher: str
    domain: str
    published_at: str | None
    retrieved_at: str
    rating: str | None = None          # fact checks: publisher's textual rating
    reviewed_claim: str | None = None  # fact checks: the claim the review is about
    claimant: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SearchProvider(abc.ABC):
    name: str
    kind: str

    @abc.abstractmethod
    def is_configured(self) -> bool: ...

    @abc.abstractmethod
    async def search(self, claim: Claim) -> list[SearchResult]:
        """Return results for one claim. Raise ProviderNotConfigured / ProviderUnavailable
        on problems. Must never return invented data."""
