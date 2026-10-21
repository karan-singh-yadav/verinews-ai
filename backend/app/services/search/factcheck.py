"""Google Fact Check Tools API (claims:search). Requires an API key (GOOGLE_FACT_CHECK_API_KEY).
Returns published ClaimReview markup from fact-checking organisations. Coverage is partial:
many claims have never been fact-checked, so an empty result means "no review found", not
"the claim is true or false"."""
from __future__ import annotations

import httpx

from backend.app.services.claims import Claim
from backend.app.services.search.base import (
    ProviderNotConfigured,
    ProviderUnavailable,
    SearchProvider,
    SearchResult,
    domain_of,
    now_iso,
    safe_url,
    sanitize_text,
)

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class FactCheckProvider(SearchProvider):
    name = "google_fact_check"
    kind = "fact_check"

    def __init__(self, api_key: str, timeout: float = 10.0, language: str | None = "en",
                 transport: httpx.AsyncBaseTransport | None = None, page_size: int = 10):
        self._key, self._timeout, self._lang = api_key, timeout, language
        self._transport, self._page_size = transport, page_size

    def is_configured(self) -> bool:
        return bool(self._key)

    async def search(self, claim: Claim) -> list[SearchResult]:
        if not self._key:
            raise ProviderNotConfigured("GOOGLE_FACT_CHECK_API_KEY is not set")
        params: dict[str, str | int] = {"query": claim.query_text(), "pageSize": self._page_size,
                                        "key": self._key}
        if self._lang:
            params["languageCode"] = self._lang
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout,
                                         trust_env=False) as client:
                resp = await client.get(ENDPOINT, params=params)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable("Fact Check API timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable("Fact Check API could not be reached") from exc

        # NOTE: never include resp.url / exception text in messages: the URL contains the key.
        if resp.status_code == 429:
            raise ProviderUnavailable("Fact Check API rate limit or quota exceeded (HTTP 429)")
        if resp.status_code in (400, 401, 403):
            raise ProviderUnavailable(
                f"Fact Check API rejected the request (HTTP {resp.status_code}); check that the "
                "API key is valid and the Fact Check Tools API is enabled for its project")
        if resp.status_code >= 500:
            raise ProviderUnavailable(f"Fact Check API error (HTTP {resp.status_code})")
        if resp.status_code != 200:
            raise ProviderUnavailable(
                f"Unexpected Fact Check API response (HTTP {resp.status_code})")
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderUnavailable("Fact Check API returned an unreadable response") from exc
        if not isinstance(data, dict):
            raise ProviderUnavailable("Fact Check API returned an unexpected payload")

        retrieved = now_iso()
        results: list[SearchResult] = []
        for c in data.get("claims") or []:
            for review in c.get("claimReview") or []:
                url = safe_url(review.get("url"))
                if not url:
                    continue
                pub = review.get("publisher") or {}
                results.append(SearchResult(
                    provider=self.name, kind=self.kind,
                    title=sanitize_text(review.get("title") or c.get("text"), 300),
                    url=url, snippet="",
                    publisher=sanitize_text(pub.get("name") or pub.get("site"), 120),
                    domain=sanitize_text(pub.get("site"), 120).lower() or domain_of(url),
                    published_at=sanitize_text(review.get("reviewDate") or c.get("claimDate"), 40)
                    or None,
                    retrieved_at=retrieved,
                    rating=sanitize_text(review.get("textualRating"), 120) or None,
                    reviewed_claim=sanitize_text(c.get("text"), 500) or None,
                    claimant=sanitize_text(c.get("claimant"), 120) or None,
                ))
        return results
