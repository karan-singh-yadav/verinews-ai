"""Evidence assessment: relevance, stance, credibility, independence, recency -> verdict.

Design rules
  * Verdicts: Supported | Contradicted | Mixed Evidence | Inconclusive.
  * No relevant evidence, failed providers, or thin evidence => Inconclusive. Absence of search
    results is NEVER treated as evidence that a claim is false.
  * Fact-check reviews (human-published ratings) outweigh news coverage.
  * News headlines can only provide WEAK signals: a headline that closely matches a claim shows
    the topic is being reported, and debunk wording in a headline suggests a rebuttal. Neither
    verifies the claim. That is why news-only verdicts need several independent domains
    (and at least one credible-tier source) and the limitations are surfaced to the user.
  * Everything shown is derived from retrieved data; explanations are generated from counts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from backend.app.core.config import Settings
from backend.app.services.claims import Claim, content_tokens
from backend.app.services.search.base import SearchResult, registered_domain

log = logging.getLogger("verinews.evidence")

SUPPORTED, CONTRADICTED, MIXED, INCONCLUSIVE = (
    "Supported", "Contradicted", "Mixed Evidence", "Inconclusive")

_MIXED_RATING = ("mixture", "mixed", "half", "partly", "partially", "misleading", "missing context",
                 "some truth", "needs context", "out of context", "outdated", "exaggerat")
_FALSE_RATING = ("false", "fake", "hoax", "pants on fire", "incorrect", "wrong", "fabricat",
                 "not true", "untrue", "inaccurate", "baseless", "no evidence", "debunk",
                 "scam", "fraud", "doctored", "manipulated", "satire")
_TRUE_RATING = ("true", "correct", "accurate", "legit", "real", "confirmed", "genuine", "verified")
_DEBUNK_CUES = re.compile(
    r"\b(debunk(?:s|ed|ing)?|hoax|fabricat\w*|fake|false claim|falsely|no evidence|"
    r"misleading|doctored|baseless|fact[- ]?check(?:ed|s)?: ?false|not true|rumou?r)\b", re.I)


@dataclass
class SourceTiers:
    fact_checker: set[str] = field(default_factory=set)
    wire_or_major: set[str] = field(default_factory=set)
    official_suffixes: tuple[str, ...] = ()

    def tier(self, domain: str) -> str:
        reg = registered_domain(domain)
        d = domain.lower()
        if d in self.fact_checker or reg in self.fact_checker:
            return "fact_checker"
        if d in self.wire_or_major or reg in self.wire_or_major:
            return "wire_or_major"
        if any(d.endswith(sfx) for sfx in self.official_suffixes):
            return "official"
        return "unrated"

    @staticmethod
    def is_credible(tier: str) -> bool:
        return tier in ("fact_checker", "wire_or_major", "official")


def load_source_tiers(path: str | Path) -> SourceTiers:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        log.warning("source tier config unreadable; all sources will be 'unrated'")
        return SourceTiers()
    tiers = raw.get("tiers") or {}
    return SourceTiers(
        fact_checker={d.lower() for d in tiers.get("fact_checker", [])},
        wire_or_major={d.lower() for d in tiers.get("wire_or_major", [])},
        official_suffixes=tuple(raw.get("official_suffixes", [])),
    )


def rating_to_stance(rating: str | None) -> str:
    """Map a fact-checker's textual rating to supports / contradicts / mixed / neutral."""
    if not rating:
        return "neutral"
    r = rating.lower()
    if any(k in r for k in _MIXED_RATING):
        return "mixed"
    if any(k in r for k in _FALSE_RATING):
        return "contradicts"
    if any(k in r for k in _TRUE_RATING):
        return "supports"
    return "neutral"


def relevance(claim: Claim, text: str) -> float:
    """0..1 lexical overlap between the claim and a retrieved title/snippet/reviewed-claim."""
    toks = set(content_tokens(text))
    ctoks = set(claim.keywords) or set(content_tokens(claim.text))
    if not ctoks or not toks:
        return 0.0
    overall = len(ctoks & toks) / len(ctoks)
    ent_tokens = {t for e in claim.entities for t in content_tokens(e)}
    if ent_tokens:
        return round(0.5 * overall + 0.5 * (len(ent_tokens & toks) / len(ent_tokens)), 3)
    return round(overall, 3)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    for candidate in (v, v[:10]):
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


@dataclass
class EvidenceItem:
    claim_index: int
    kind: str
    provider: str
    title: str
    url: str
    snippet: str
    publisher: str
    domain: str
    published_at: str | None
    retrieved_at: str
    rating: str | None
    reviewed_claim: str | None
    relevance: float
    stance: str
    stance_basis: str
    source_tier: str
    stale: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _build_item(idx: int, claim: Claim, r: SearchResult, rel: float, tiers: SourceTiers,
                s: Settings, now: datetime) -> EvidenceItem | None:
    tier = tiers.tier(r.domain)
    stale = False
    if r.kind == "fact_check":
        if rel < s.factcheck_relevance_threshold:
            return None
        stance = rating_to_stance(r.rating)
        basis = (f"Publisher rating: '{r.rating}'" if r.rating
                 else "Fact-check without a usable rating")
    else:
        if rel < s.relevance_threshold:
            return None
        dt = _parse_date(r.published_at)
        stale = bool(dt and (now - dt).days > s.news_max_age_days)
        if stale:
            stance, basis = "neutral", "Older than the recency window; not counted"
        elif _DEBUNK_CUES.search(f"{r.title} {r.snippet}"):
            stance = "contradicts"
            basis = ("Headline contains debunk/rebuttal wording "
                     "(keyword heuristic, not verification)")
        elif rel >= s.support_relevance_threshold:
            stance = "supports"
            basis = ("Headline closely matches the claim and has no debunk wording "
                     "(topical corroboration, not verification)")
        else:
            stance, basis = "neutral", "Related coverage; headline overlap too weak to count"
    return EvidenceItem(
        claim_index=idx, kind=r.kind, provider=r.provider, title=r.title, url=r.url,
        snippet=r.snippet, publisher=r.publisher, domain=r.domain, published_at=r.published_at,
        retrieved_at=r.retrieved_at, rating=r.rating, reviewed_claim=r.reviewed_claim,
        relevance=rel, stance=stance, stance_basis=basis, source_tier=tier, stale=stale)


def _distinct(items: list[EvidenceItem], stance: str, kind: str | None = None) -> dict[str, str]:
    """registered domain -> tier, for distinct domains holding `stance` (source independence)."""
    out: dict[str, str] = {}
    for it in items:
        if it.stance == stance and (kind is None or it.kind == kind):
            out.setdefault(registered_domain(it.domain), it.source_tier)
    return out


def assess_claim(items: list[EvidenceItem], s: Settings) -> tuple[str, str]:
    fc_sup = _distinct(items, "supports", "fact_check")
    fc_con = _distinct(items, "contradicts", "fact_check")
    fc_mix = _distinct(items, "mixed", "fact_check")
    n_sup, n_con = _distinct(items, "supports", "news"), _distinct(items, "contradicts", "news")
    credible = SourceTiers.is_credible

    if fc_sup or fc_con or fc_mix:
        if fc_mix or (fc_sup and fc_con):
            return MIXED, (f"Fact-checkers disagree or rate it partly true "
                           f"({len(fc_sup)} supporting, {len(fc_con)} contradicting, "
                           f"{len(fc_mix)} mixed/misleading).")
        if fc_con:
            if len(n_sup) >= s.min_independent_sources:
                return MIXED, ("Fact-check(s) contradict the claim, but several news outlets "
                               "report it without rebuttal wording.")
            return CONTRADICTED, (f"{len(fc_con)} independent fact-checking source(s) rated "
                                  "the claim false or unsupported.")
        if len(n_con) >= 2:
            return MIXED, ("Fact-check(s) support the claim, but several news outlets "
                           "carry rebuttal wording.")
        return SUPPORTED, f"{len(fc_sup)} independent fact-checking source(s) rated the claim true."

    # news-only evidence (weak signals)
    enough_sup = (len(n_sup) >= s.min_independent_sources
                  and any(credible(t) for t in n_sup.values()))
    enough_con = (len(n_con) >= s.min_debunk_sources
                  and any(credible(t) for t in n_con.values()))
    if enough_sup and enough_con:
        return MIXED, "News coverage both repeats the claim and carries rebuttal wording."
    if enough_con:
        return CONTRADICTED, (f"{len(n_con)} independent outlets published headlines with "
                              "rebuttal wording about this claim (headline-level signal only).")
    if enough_sup:
        return SUPPORTED, (f"{len(n_sup)} independent outlets published closely matching "
                           "coverage with no rebuttal wording (headline-level signal only).")
    if n_sup or n_con:
        return INCONCLUSIVE, ("Some related coverage exists but it is too thin, too few "
                              "independent domains, or lacks a credible-tier source.")
    return INCONCLUSIVE, ("No sufficiently relevant fact-checks or news reports were found. "
                          "A lack of results is not evidence that the claim is false.")


LIMITATIONS = [
    "Evidence comes only from the providers that responded; coverage is partial.",
    "Relevance and stance are keyword/overlap heuristics on titles and ratings, not a reading "
    "of full articles.",
    "News headlines can show a topic is reported, not that a claim is accurate.",
    "Source tiers are an editable heuristic list; 'unrated' does not mean unreliable.",
    "Always open the cited sources and judge for yourself.",
]


def assess(claims: list[Claim], results: dict[int, list[SearchResult]], s: Settings,
           tiers: SourceTiers, now: datetime | None = None) -> dict:
    """Combine per-claim evidence into an overall structured assessment."""
    now = now or datetime.now(UTC)
    per_claim = []
    discarded = 0
    for idx, claim in enumerate(claims):
        items: list[EvidenceItem] = []
        seen_urls: set[str] = set()  # de-duplicate per claim: one article may address several
        for r in results.get(idx, []):
            if r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            text = " ".join(filter(None, [r.reviewed_claim, r.title, r.snippet]))
            item = _build_item(idx, claim, r, relevance(claim, text), tiers, s, now)
            if item is None:
                discarded += 1
            else:
                items.append(item)
        verdict, reason = assess_claim(items, s)
        items.sort(key=lambda i: (i.kind != "fact_check", -i.relevance))
        per_claim.append({"claim": claim.to_dict(), "verdict": verdict, "reason": reason,
                          "evidence": [i.to_dict() for i in items]})

    verdicts = [c["verdict"] for c in per_claim]
    if not verdicts:
        overall, why = INCONCLUSIVE, "No checkable claims were found in the text."
    elif CONTRADICTED in verdicts and SUPPORTED in verdicts:
        overall, why = MIXED, "Some claims are supported while others are contradicted."
    elif CONTRADICTED in verdicts:
        n = verdicts.count(CONTRADICTED)
        overall, why = CONTRADICTED, f"{n} of {len(verdicts)} checked claim(s) were contradicted."
    elif MIXED in verdicts:
        overall, why = MIXED, "Evidence for at least one claim is mixed or conflicting."
    elif all(v == SUPPORTED for v in verdicts):
        overall, why = SUPPORTED, "All checked claims were supported by the retrieved evidence."
    elif SUPPORTED in verdicts:
        overall, why = INCONCLUSIVE, (
            f"Only {verdicts.count(SUPPORTED)} of {len(verdicts)} claims could be corroborated; "
            "the rest had insufficient evidence.")
    else:
        overall, why = INCONCLUSIVE, "Insufficient or no relevant evidence was found."

    all_items = [e for c in per_claim for e in c["evidence"]]
    return {
        "verdict": overall,
        "summary": why,
        "claims": per_claim,
        "supporting": [e for e in all_items if e["stance"] == "supports"],
        "contradicting": [e for e in all_items if e["stance"] == "contradicts"],
        "other": [e for e in all_items if e["stance"] in ("mixed", "neutral")],
        "evidence_count": len(all_items),
        "discarded_irrelevant": discarded,
        "limitations": LIMITATIONS,
    }
