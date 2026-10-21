import asyncio
from datetime import UTC, datetime

import pytest

from backend.app.core.config import Settings
from backend.app.services.claims import Claim
from backend.app.services.evidence import (
    CONTRADICTED,
    INCONCLUSIVE,
    MIXED,
    SUPPORTED,
    SourceTiers,
    assess,
    load_source_tiers,
    rating_to_stance,
    relevance,
)
from backend.app.services.search.base import ProviderUnavailable
from backend.app.services.verifier import Verifier
from tests.helpers import StubProvider, result

S = Settings(_env_file=None)
TIERS = SourceTiers(fact_checker={"altnews.in", "snopes.com"},
                    wire_or_major={"reuters.com", "bbc.co.uk", "thehindu.com"},
                    official_suffixes=(".gov.in",))
CLAIM = Claim(text="The ministry announced 4,500 new clinics will open in Uttar Pradesh",
              entities=["Uttar Pradesh"], keywords=["ministry", "announc", "4,500", "new", "clinic",
                                                    "open", "uttar", "pradesh"])
NOW = datetime(2026, 10, 16, tzinfo=UTC)
MATCH = "Ministry announced 4,500 new clinics to open in Uttar Pradesh"


def run_assess(items, claims=None):
    claims = claims or [CLAIM]
    return assess(claims, {0: items}, S, TIERS, now=NOW)


def fcheck(domain, rating, claim_text=MATCH):
    return result(kind="fact_check", title="Fact check", url=f"https://{domain}/fc",
                  domain=domain, rating=rating, reviewed_claim=claim_text)


def news(domain, title=MATCH, date="2026-10-01T00:00:00Z"):
    return result(title=title, url=f"https://{domain}/n", domain=domain, published_at=date)


@pytest.mark.parametrize("rating,stance", [
    ("False", "contradicts"), ("Pants on Fire", "contradicts"), ("Not true", "contradicts"),
    ("Mostly False", "contradicts"), ("True", "supports"), ("Mostly True", "supports"),
    ("Half True", "mixed"), ("Misleading", "mixed"), ("Partly false", "mixed"),
    ("Unproven", "neutral"), (None, "neutral"), ("", "neutral")])
def test_rating_mapping(rating, stance):
    assert rating_to_stance(rating) == stance


def test_relevance_scores_overlap():
    assert relevance(CLAIM, MATCH) > 0.8
    assert relevance(CLAIM, "Cricket team wins the world cup final") < 0.2


def test_no_results_is_inconclusive_and_never_fake():
    out = run_assess([])
    assert out["verdict"] == INCONCLUSIVE
    assert "not evidence that the claim is false" in out["claims"][0]["reason"]


def test_fact_check_false_is_contradicted_with_citation():
    out = run_assess([fcheck("altnews.in", "False")])
    assert out["verdict"] == CONTRADICTED
    ev = out["contradicting"][0]
    assert ev["url"] == "https://altnews.in/fc" and ev["rating"] == "False"
    assert ev["source_tier"] == "fact_checker" and ev["retrieved_at"]


def test_fact_check_true_is_supported():
    assert run_assess([fcheck("snopes.com", "True")])["verdict"] == SUPPORTED


def test_conflicting_fact_checks_are_mixed():
    out = run_assess([fcheck("snopes.com", "True"), fcheck("altnews.in", "False")])
    assert out["verdict"] == MIXED


def test_half_true_is_mixed():
    assert run_assess([fcheck("snopes.com", "Half True")])["verdict"] == MIXED


def test_irrelevant_fact_check_is_discarded():
    out = run_assess([fcheck("snopes.com", "False", claim_text="Aliens built the pyramids")])
    assert out["verdict"] == INCONCLUSIVE and out["discarded_irrelevant"] == 1


def test_news_needs_independent_domains_and_a_credible_source():
    three = [news("reuters.com"), news("bbc.co.uk"), news("blog-a.com")]
    assert run_assess(three)["verdict"] == SUPPORTED
    same_site = [news("reuters.com"), news("www.reuters.com"), news("reuters.com", MATCH + " today")]
    same_site[2].url = "https://reuters.com/other"
    assert run_assess(same_site)["verdict"] == INCONCLUSIVE        # one registered domain
    no_credible = [news("blog-a.com"), news("blog-b.com"), news("blog-c.com")]
    assert run_assess(no_credible)["verdict"] == INCONCLUSIVE
    assert run_assess(three[:2])["verdict"] == INCONCLUSIVE        # too few domains


def test_news_with_debunk_wording_contradicts():
    items = [news("reuters.com", "Debunked: " + MATCH), news("thehindu.com", MATCH + " is a hoax")]
    out = run_assess(items)
    assert out["verdict"] == CONTRADICTED
    assert "heuristic" in out["contradicting"][0]["stance_basis"].lower()


def test_stale_news_is_not_counted():
    old = "2020-01-01T00:00:00Z"
    out = run_assess([news("reuters.com", date=old), news("bbc.co.uk", date=old),
                      news("thehindu.com", date=old)])
    assert out["verdict"] == INCONCLUSIVE
    assert all(e["stale"] for e in out["other"])


def test_fact_check_outweighs_news_but_strong_conflict_downgrades_to_mixed():
    news3 = [news("reuters.com"), news("bbc.co.uk"), news("thehindu.com")]
    assert run_assess([fcheck("altnews.in", "False"), *news3])["verdict"] == MIXED
    assert run_assess([fcheck("altnews.in", "False")])["verdict"] == CONTRADICTED


def test_multi_claim_aggregation():
    c2 = Claim(text="Officials arrested 12 protesters in Mumbai", entities=["Mumbai"],
               keywords=["official", "arrest", "protester", "mumbai", "12"])
    mk = lambda items: assess([CLAIM, c2], {0: items[0], 1: items[1]}, S, TIERS, now=NOW)  # noqa: E731
    bad = [fcheck("altnews.in", "False")]
    good = [fcheck("snopes.com", "True", "Officials arrested 12 protesters in Mumbai")]
    assert mk([bad, good])["verdict"] == MIXED
    assert mk([bad, []])["verdict"] == CONTRADICTED
    assert mk([[fcheck("snopes.com", "True")], []])["verdict"] == INCONCLUSIVE
    assert mk([[fcheck("snopes.com", "True")], good])["verdict"] == SUPPORTED


def test_duplicate_urls_are_collapsed():
    a = news("reuters.com")
    out = run_assess([a, a, a])
    assert out["evidence_count"] == 1


def test_source_tier_config_loading(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("tiers:\n  fact_checker: [Example-FC.org]\nofficial_suffixes: ['.gov']\n")
    t = load_source_tiers(p)
    assert t.tier("example-fc.org") == "fact_checker" and t.tier("data.nasa.gov") == "official"
    assert t.tier("random.com") == "unrated"
    assert load_source_tiers(tmp_path / "missing.yaml").tier("reuters.com") == "unrated"


# ---------------- verifier orchestration ----------------
TEXT = ("The health ministry announced that 4,500 new clinics will open in Uttar Pradesh next "
        "year, according to officials in Lucknow.")


def verify(providers, **settings):
    s = Settings(_env_file=None, **settings)
    return asyncio.run(Verifier(providers, s, TIERS).verify(TEXT))


def test_no_configured_provider_means_incomplete_inconclusive():
    out = verify([StubProvider("google_fact_check", "fact_check", configured=False)])
    assert out["verdict"] == INCONCLUSIVE and out["completed"] is False
    assert "could not be completed" in out["summary"]
    assert out["providers"][0]["status"] == "not_configured"


def test_provider_failure_is_reported_not_hidden():
    out = verify([StubProvider(error=ProviderUnavailable("rate limit hit"))])
    assert out["completed"] is False and out["verdict"] == INCONCLUSIVE
    assert out["providers"][0]["status"] == "error"
    assert "rate limit hit" in out["summary"]


def test_ok_provider_with_no_results_is_completed_but_inconclusive():
    out = verify([StubProvider(results=[])])
    assert out["completed"] is True and out["verdict"] == INCONCLUSIVE
    assert "not evidence" in out["claims"][0]["reason"]


def test_one_failing_one_working_provider_still_completes():
    ok = StubProvider("p_ok", "fact_check", results=[fcheck("altnews.in", "False")])
    bad = StubProvider("p_bad", error=ProviderUnavailable("down"))
    out = verify([ok, bad])
    assert out["completed"] is True and out["verdict"] == CONTRADICTED
    assert {p["name"]: p["status"] for p in out["providers"]} == {"p_ok": "ok", "p_bad": "error"}


def test_provider_crash_is_contained_and_not_leaked():
    out = verify([StubProvider(crash=True)])
    assert out["completed"] is False
    assert "boom" not in str(out) and "secret internals" not in str(out)


def test_verification_timeout_yields_inconclusive():
    out = verify([StubProvider(delay=2.0)], verification_timeout_seconds=0.05)
    assert out["completed"] is False and out["verdict"] == INCONCLUSIVE
    assert "timed out" in out["summary"]


def test_text_without_claims_is_incomplete():
    s = Settings(_env_file=None)
    out = asyncio.run(Verifier([StubProvider()], s, TIERS).verify("   "))
    assert out["verdict"] == INCONCLUSIVE and out["completed"] is False


def test_results_are_never_invented():
    r = fcheck("altnews.in", "False")
    out = verify([StubProvider("fc", "fact_check", results=[r])])
    urls = {e["url"] for e in out["contradicting"] + out["supporting"] + out["other"]}
    assert urls <= {r.url}
