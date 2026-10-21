import asyncio

import httpx
import pytest

from backend.app.services.claims import Claim
from backend.app.services.search.base import (
    ProviderNotConfigured,
    ProviderUnavailable,
    registered_domain,
    safe_url,
    sanitize_text,
)
from backend.app.services.search.factcheck import FactCheckProvider
from backend.app.services.search.gdelt import GdeltProvider, build_gdelt_query, parse_seendate

run = asyncio.run
KEY = "SECRET-KEY-123"
CLAIM = Claim(text="The ministry announced 4,500 new clinics in Uttar Pradesh",
              entities=["Uttar Pradesh"], keywords=["ministry", "announc", "clinic", "4,500"])


def fc(handler, key=KEY):
    return FactCheckProvider(key, transport=httpx.MockTransport(handler))


def test_factcheck_without_key_reports_not_configured_and_makes_no_request():
    calls = []
    p = fc(lambda r: calls.append(r) or httpx.Response(200, json={}), key="")
    assert not p.is_configured()
    with pytest.raises(ProviderNotConfigured, match="GOOGLE_FACT_CHECK_API_KEY"):
        run(p.search(CLAIM))
    assert calls == []


def test_factcheck_parses_claim_reviews_and_records_retrieval_time():
    payload = {"claims": [{"text": "4,500 clinics will open", "claimant": "Someone",
                           "claimDate": "2026-03-12T00:00:00Z",
                           "claimReview": [{"publisher": {"name": "Alt News", "site": "altnews.in"},
                                            "url": "https://www.altnews.in/x", "title": "Fact check",
                                            "textualRating": "False", "reviewDate": "2026-03-14T00:00:00Z"},
                                           {"url": "javascript:alert(1)", "title": "bad"}]}]}
    seen = {}

    def handler(req: httpx.Request):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json=payload)
    out = run(fc(handler).search(CLAIM))
    assert len(out) == 1                                # javascript: link dropped
    r = out[0]
    assert r.kind == "fact_check" and r.rating == "False" and r.domain == "altnews.in"
    assert r.url == "https://www.altnews.in/x" and r.retrieved_at and r.published_at
    assert seen["key"] == KEY and seen["pageSize"] == "10"


def test_factcheck_empty_result_is_empty_list_not_error():
    assert run(fc(lambda r: httpx.Response(200, json={})).search(CLAIM)) == []


@pytest.mark.parametrize("status,needle", [(429, "rate limit"), (403, "rejected"),
                                           (400, "rejected"), (500, "error"), (302, "Unexpected")])
def test_factcheck_http_errors_never_leak_the_api_key(status, needle):
    with pytest.raises(ProviderUnavailable) as ei:
        run(fc(lambda r: httpx.Response(status)).search(CLAIM))
    assert needle in str(ei.value) and KEY not in str(ei.value)


def test_factcheck_timeout_network_and_bad_json():
    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)

    def netfail(r):
        raise httpx.ConnectError("down", request=r)
    for h, needle in ((timeout, "timed out"), (netfail, "could not be reached")):
        with pytest.raises(ProviderUnavailable, match=needle) as ei:
            run(fc(h).search(CLAIM))
        assert KEY not in str(ei.value)
    with pytest.raises(ProviderUnavailable, match="unreadable"):
        run(fc(lambda r: httpx.Response(200, text="<html>nope")).search(CLAIM))
    with pytest.raises(ProviderUnavailable, match="unexpected payload"):
        run(fc(lambda r: httpx.Response(200, json=[1, 2])).search(CLAIM))


def gd(handler, **kw):
    return GdeltProvider(min_interval=0, transport=httpx.MockTransport(handler), **kw)


def test_gdelt_parses_articles():
    payload = {"articles": [{"url": "https://www.thehindu.com/a", "title": "Clinics <b>open</b>",
                             "domain": "thehindu.com", "seendate": "20260312T101500Z"},
                            {"url": "ftp://bad.example/x", "title": "skip me", "domain": "bad"}]}
    out = run(gd(lambda r: httpx.Response(200, json=payload)).search(CLAIM))
    assert len(out) == 1 and out[0].title == "Clinics open" and out[0].kind == "news"
    assert out[0].published_at == "2026-03-12T10:15:00Z" and out[0].domain == "thehindu.com"


def test_gdelt_throttle_text_is_an_error_not_empty_results():
    h = lambda r: httpx.Response(200, text="Please limit requests to one every 5 seconds.")  # noqa: E731
    with pytest.raises(ProviderUnavailable, match="rate limit"):
        run(gd(h).search(CLAIM))
    with pytest.raises(ProviderUnavailable, match="rate limit"):
        run(gd(lambda r: httpx.Response(429)).search(CLAIM))


def test_gdelt_non_json_error_body_and_empty_body():
    with pytest.raises(ProviderUnavailable, match="did not return JSON"):
        run(gd(lambda r: httpx.Response(200, text="The specified phrase is too short.")).search(CLAIM))
    assert run(gd(lambda r: httpx.Response(200, text="")).search(CLAIM)) == []


def test_gdelt_server_error_and_timeout():
    with pytest.raises(ProviderUnavailable, match="HTTP 503"):
        run(gd(lambda r: httpx.Response(503)).search(CLAIM))

    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)
    with pytest.raises(ProviderUnavailable, match="timed out"):
        run(gd(timeout).search(CLAIM))


def test_gdelt_query_building():
    q = build_gdelt_query(Claim(text="x", entities=["Narendra Modi", "AI"], keywords=["visit", "12"]))
    assert '"Narendra Modi"' in q and "visit" in q and "12" not in q
    assert build_gdelt_query(Claim(text="x")) == ""


def test_helpers():
    assert parse_seendate("garbage") is None
    assert safe_url("javascript:alert(1)") is None and safe_url("https://a.com/x")
    assert sanitize_text("<script>x</script>Hello\x00 &amp; bye") == "x Hello & bye"
    assert registered_domain("news.bbc.co.uk") == "bbc.co.uk"
    assert registered_domain("www.altnews.in") == "altnews.in"
