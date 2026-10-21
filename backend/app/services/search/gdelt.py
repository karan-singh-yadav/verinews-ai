"""GDELT DOC 2.0 article search (keyless). Returns news article METADATA (title, url, domain,
first-seen time) from a rolling window of roughly the last three months. GDELT asks for about
one request per 5 seconds per IP and signals throttling with a plain-text body, so this client
paces itself and treats non-JSON bodies as an error, never as "no results"."""
from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime

import httpx

from backend.app.services.claims import Claim
from backend.app.services.search.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResult,
    domain_of,
    now_iso,
    safe_url,
    sanitize_text,
)

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_SAFE_TERM = re.compile(r"[^A-Za-z0-9\u00C0-\u024F\u0900-\u097F' \-]")


def build_gdelt_query(claim: Claim) -> str:
    """Quoted multi-word entities + a few distinctive keywords, AND-ed (GDELT's default)."""
    terms: list[str] = []
    for ent in claim.entities[:2]:
        clean = _SAFE_TERM.sub(" ", ent).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) < 3:
            continue
        terms.append(f'"{clean}"' if " " in clean else clean)
    ent_words = {w.lower() for e in claim.entities for w in e.split()}
    for kw in claim.keywords:
        if len(terms) >= 4:
            break
        if len(kw) >= 4 and kw.lower() not in ent_words and not kw.isdigit():
            terms.append(_SAFE_TERM.sub("", kw))
    return " ".join(t for t in terms if t)[:250]


def parse_seendate(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").isoformat() + "Z"
    except ValueError:
        return None


class GdeltProvider(SearchProvider):
    name = "gdelt"
    kind = "news"

    def __init__(self, enabled: bool = True, timeout: float = 15.0, min_interval: float = 5.5,
                 transport: httpx.AsyncBaseTransport | None = None, max_records: int = 20):
        self._enabled, self._timeout, self._min_interval = enabled, timeout, min_interval
        self._transport, self._max = transport, max_records
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    def is_configured(self) -> bool:
        return self._enabled

    async def search(self, claim: Claim) -> list[SearchResult]:
        query = build_gdelt_query(claim)
        if not query:
            return []
        params = {"query": query, "mode": "artlist", "format": "json",
                  "maxrecords": self._max, "sort": "hybridrel", "timespan": "3months"}
        async with self._lock:  # one request at a time, spaced out
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout,
                                             trust_env=False) as client:
                    resp = await client.get(ENDPOINT, params=params)
            except httpx.TimeoutException as exc:
                raise ProviderUnavailable("GDELT timed out") from exc
            except httpx.HTTPError as exc:
                raise ProviderUnavailable("GDELT could not be reached") from exc
            finally:
                self._last_call = time.monotonic()

        text = resp.text or ""
        if resp.status_code == 429 or "limit requests" in text.lower():
            raise ProviderUnavailable("GDELT rate limit hit (about 1 request / 5 s allowed)")
        if resp.status_code >= 500:
            raise ProviderUnavailable(f"GDELT error (HTTP {resp.status_code})")
        if resp.status_code != 200:
            raise ProviderUnavailable(f"Unexpected GDELT response (HTTP {resp.status_code})")
        if not text.strip():
            return []  # GDELT returns an empty body when nothing matches
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderUnavailable(
                f"GDELT did not return JSON: {sanitize_text(text, 120)}") from exc

        retrieved = now_iso()
        out: list[SearchResult] = []
        for a in (data.get("articles") or []) if isinstance(data, dict) else []:
            url = safe_url(a.get("url"))
            title = sanitize_text(a.get("title"), 300)
            if not url or not title:
                continue
            dom = sanitize_text(a.get("domain"), 120).lower() or domain_of(url)
            out.append(SearchResult(
                provider=self.name, kind=self.kind, title=title, url=url, snippet="",
                publisher=dom, domain=dom, published_at=parse_seendate(a.get("seendate")),
                retrieved_at=retrieved,
            ))
        return out
