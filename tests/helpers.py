"""Test doubles. These live ONLY in tests; the application never fabricates search results."""
from __future__ import annotations

import asyncio

from backend.app.services.claims import Claim
from backend.app.services.search.base import (
    ProviderNotConfigured,
    SearchProvider,
    SearchResult,
)


class StubProvider(SearchProvider):
    def __init__(self, name="stub", kind="news", results=None, configured=True, error=None,
                 delay=0.0, crash=False):
        self.name, self.kind = name, kind
        self.results, self.configured, self.error = results or [], configured, error
        self.delay, self.crash, self.calls = delay, crash, 0

    def is_configured(self) -> bool:
        return self.configured

    async def search(self, claim: Claim) -> list[SearchResult]:
        self.calls += 1
        if not self.configured:
            raise ProviderNotConfigured("not configured")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.crash:
            raise RuntimeError("boom secret internals")
        if self.error:
            raise self.error
        return list(self.results)


def result(kind="news", title="t", url="https://example.org/a", domain="example.org",
           rating=None, reviewed_claim=None, published_at=None, provider="stub") -> SearchResult:
    return SearchResult(provider=provider, kind=kind, title=title, url=url, snippet="",
                        publisher=domain, domain=domain, published_at=published_at,
                        retrieved_at="2026-10-16T00:00:00+00:00", rating=rating,
                        reviewed_claim=reviewed_claim)
