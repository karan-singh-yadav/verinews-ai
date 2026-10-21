"""Orchestrates online verification: claims -> search providers -> evidence assessment.

Never raises for provider problems. If verification cannot be completed the result is an
explicit Inconclusive with completed=False and the reason for every provider."""
from __future__ import annotations

import asyncio
import logging

from backend.app.core.config import Settings
from backend.app.services.claims import Claim, extract_claims
from backend.app.services.evidence import (
    INCONCLUSIVE,
    LIMITATIONS,
    SourceTiers,
    assess,
)
from backend.app.services.search.base import (
    ProviderError,
    SearchProvider,
    SearchResult,
    now_iso,
)

log = logging.getLogger("verinews.verifier")


class Verifier:
    def __init__(self, providers: list[SearchProvider], settings: Settings, tiers: SourceTiers):
        self.providers, self.settings, self.tiers = providers, settings, tiers

    def provider_overview(self) -> dict[str, bool]:
        return {p.name: p.is_configured() for p in self.providers}

    async def _run_provider(self, provider: SearchProvider, claims: list[Claim]):
        """Returns (status_dict, {claim_index: results})."""
        status = {"name": provider.name, "kind": provider.kind, "status": "ok",
                  "results": 0, "error": None}
        out: dict[int, list[SearchResult]] = {}
        if not provider.is_configured():
            status.update(status="not_configured", error=_not_configured_msg(provider))
            return status, out
        failures: list[str] = []
        for idx, claim in enumerate(claims):
            try:
                found = await provider.search(claim)
                out[idx] = found
                status["results"] += len(found)
            except ProviderError as exc:
                failures.append(str(exc))
            except Exception:  # defensive: a provider bug must not break the request
                log.exception("provider crashed", extra={"provider": provider.name})
                failures.append("Unexpected provider error")
        if failures:
            ok_any = len(failures) < len(claims)
            status.update(status="partial" if ok_any else "error", error=failures[0])
        return status, out

    async def verify(self, text: str) -> dict:
        s = self.settings
        started = now_iso()
        claims = extract_claims(text, max_claims=s.max_claims_searched)
        base = {"started_at": started, "providers": [], "claims_checked": len(claims)}
        if not claims:
            return {**base, "verdict": INCONCLUSIVE, "completed": False,
                    "summary": "No checkable factual claims could be extracted from the text.",
                    "claims": [], "supporting": [], "contradicting": [], "other": [],
                    "evidence_count": 0, "discarded_irrelevant": 0, "limitations": LIMITATIONS,
                    "completed_at": now_iso()}

        async def guarded(p: SearchProvider):
            try:
                return await self._run_provider(p, claims)
            except Exception:  # pragma: no cover - safety net
                log.exception("provider wrapper failed")
                return ({"name": p.name, "kind": p.kind, "status": "error", "results": 0,
                         "error": "Unexpected provider error"}, {})

        try:
            runs = await asyncio.wait_for(
                asyncio.gather(*(guarded(p) for p in self.providers)),
                timeout=s.verification_timeout_seconds)
        except TimeoutError:
            runs = [({"name": p.name, "kind": p.kind, "status": "error", "results": 0,
                      "error": "Verification timed out"}, {}) for p in self.providers]

        statuses = [r[0] for r in runs]
        merged: dict[int, list[SearchResult]] = {}
        for _, per_claim in runs:
            for idx, found in per_claim.items():
                merged.setdefault(idx, []).extend(found)

        result = assess(claims, merged, s, self.tiers)
        usable = [st for st in statuses if st["status"] in ("ok", "partial")]
        completed = bool(usable)
        if not completed:
            reasons = "; ".join(f"{st['name']}: {st['error']}" for st in statuses) or \
                "no search providers are configured"
            result["verdict"] = INCONCLUSIVE
            result["summary"] = f"Online verification could not be completed ({reasons})."
        elif any(st["status"] in ("error", "partial", "not_configured") for st in statuses) \
                and result["verdict"] == INCONCLUSIVE:
            result["summary"] += " Some providers were unavailable, so coverage was reduced."
        return {**base, **result, "providers": statuses, "completed": completed,
                "completed_at": now_iso()}


def _not_configured_msg(provider: SearchProvider) -> str:
    if provider.name == "google_fact_check":
        return "GOOGLE_FACT_CHECK_API_KEY is not set"
    return "Provider is disabled or not configured"
