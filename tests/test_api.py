import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.evidence import SourceTiers
from backend.app.services.search.base import ProviderUnavailable
from backend.app.services.verifier import Verifier
from ml.inference.predictor import Prediction, Predictor
from tests.helpers import StubProvider, result

UNRELATED = ("simmer the onions with garlic then stir in tomato paste basil and oregano "
             "before folding the pasta into the warm skillet with olive oil and parmesan cheese")
CLAIM_TEXT = ("The health ministry announced that 4,500 new clinics will open in Uttar Pradesh "
              "next year, according to officials in Lucknow.")
PUBLIC = "93.184.216.34"


class FakePredictor:
    """Deterministic predictor stand-in so routing logic can be tested independent of ML."""
    version, metadata = "fake-1", {"model_version": "fake-1"}

    def __init__(self, needs_verification=False):
        self.needs = needs_verification

    def predict(self, text):
        return Prediction(label="real", fake_probability=0.1, confidence=0.9,
                          confidence_validated=True, low_confidence=self.needs,
                          novelty_score=0.1, novelty_threshold=0.5, out_of_distribution=False,
                          needs_verification=self.needs,
                          verification_reasons=["forced by test"] if self.needs else [],
                          model_version="fake-1")


@pytest.fixture
def make_client(tmp_path):
    def _make(predictor=None, providers=None, verifier=None, transport=None, **overrides):
        cfg = {"feedback_path": str(tmp_path / "fb.jsonl"), "model_dir": str(tmp_path / "no-models"),
               "rate_limit_per_minute": 1000, "gdelt_min_interval_seconds": 0,
               "monitoring_log_path": str(tmp_path / "events.jsonl"), **overrides}
        s = Settings(_env_file=None, **cfg)
        v = verifier or Verifier(providers if providers is not None else [StubProvider()],
                                 s, SourceTiers())

        async def resolver(host, port):
            return [PUBLIC]
        app = create_app(s, predictor=predictor, verifier=v, resolver=resolver,
                         fetch_transport=transport)
        return TestClient(app)
    return _make


@pytest.fixture
def real_predictor(trained_models):
    return Predictor.load(trained_models[0])


# ---------- health / model-info ----------
def test_health_degraded_without_model_and_ok_with_model(make_client, real_predictor):
    r = make_client().get("/api/health").json()
    assert r["status"] == "degraded" and r["model_loaded"] is False
    r = make_client(predictor=real_predictor).get("/api/health").json()
    assert r["status"] == "ok" and r["model_version"] == "vtest"
    assert set(r["providers_configured"]) == {"stub"}


def test_model_info_reports_recorded_metadata_only(make_client, real_predictor):
    c = make_client(predictor=real_predictor)
    info = c.get("/api/model-info").json()
    assert info["synthetic_data"] is True and info["model_version"] == "vtest"
    assert "confusion_matrix" in info["test_metrics"] and info["limitations"]
    assert make_client().get("/api/model-info").status_code == 503


# ---------- predict ----------
def test_predict_ok_and_headers(make_client, real_predictor):
    r = make_client(predictor=real_predictor).post("/api/predict", json={"text": CLAIM_TEXT})
    assert r.status_code == 200
    body = r.json()
    p = body["prediction"]
    assert p["label"] in ("real", "fake") and 0 <= p["fake_probability"] <= 1
    assert "not proof" in body["disclaimer"]
    assert r.headers["x-request-id"] == body["request_id"]
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("payload", [
    {}, {"text": ""}, {"text": "   "}, {"text": 123}, {"text": None}, {"text": "a" * 20001},
    {"text": "<b></b>"},
])
def test_predict_rejects_invalid_input_without_echoing_it(make_client, real_predictor, payload):
    r = make_client(predictor=real_predictor).post("/api/predict", json=payload)
    assert r.status_code == 422
    assert "aaaa" not in r.text and "Traceback" not in r.text


def test_predict_non_json_body_is_422(make_client, real_predictor):
    r = make_client(predictor=real_predictor).post(
        "/api/predict", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_predict_without_model_is_503(make_client):
    r = make_client().post("/api/predict", json={"text": CLAIM_TEXT})
    assert r.status_code == 503 and r.json()["error"] == "http_503"


def test_oversized_body_is_413(make_client, real_predictor):
    r = make_client(predictor=real_predictor).post("/api/predict", json={"text": "a" * 250_000})
    assert r.status_code == 413


# ---------- verify ----------
def test_unfamiliar_text_triggers_verification_with_reasons(make_client, real_predictor):
    ev = result(kind="fact_check", title="Check", url="https://snopes.com/x", domain="snopes.com",
                rating="False", reviewed_claim="simmer onions garlic tomato paste basil pasta")
    c = make_client(predictor=real_predictor,
                    providers=[StubProvider("fc", "fact_check", results=[ev])])
    r = c.post("/api/verify", json={"text": UNRELATED})
    assert r.status_code == 200
    v = r.json()["verification"]
    assert v["triggered"] and any("unfamiliar" in x for x in v["reasons"])
    assert v["result"]["completed"] is True
    cited = {e["url"] for e in v["result"]["contradicting"] + v["result"]["other"]}
    assert cited <= {"https://snopes.com/x"}               # only retrieved URLs ever appear


def test_confident_familiar_text_does_not_trigger_unless_forced(make_client):
    stub = StubProvider()
    c = make_client(predictor=FakePredictor(False), providers=[stub])
    body = c.post("/api/verify", json={"text": CLAIM_TEXT}).json()
    assert body["verification"]["triggered"] is False and stub.calls == 0
    assert body["verification"]["result"] is None and "not proof" in body["verification"]["note"]
    forced = c.post("/api/verify", json={"text": CLAIM_TEXT, "force_verification": True}).json()
    assert forced["verification"]["triggered"] and stub.calls > 0


def test_uncertain_prediction_triggers_verification(make_client):
    c = make_client(predictor=FakePredictor(True))
    v = c.post("/api/verify", json={"text": CLAIM_TEXT}).json()["verification"]
    assert v["triggered"] and v["reasons"] == ["forced by test"]


def test_verify_with_unconfigured_providers_reports_incomplete(make_client):
    c = make_client(predictor=FakePredictor(True),
                    providers=[StubProvider("google_fact_check", "fact_check", configured=False)])
    res = c.post("/api/verify", json={"text": CLAIM_TEXT}).json()["verification"]["result"]
    assert res["verdict"] == "Inconclusive" and res["completed"] is False
    assert "could not be completed" in res["summary"]


def test_verify_with_failing_provider_is_inconclusive_not_error(make_client):
    c = make_client(predictor=FakePredictor(True),
                    providers=[StubProvider(error=ProviderUnavailable("quota exceeded"))])
    r = c.post("/api/verify", json={"text": CLAIM_TEXT})
    assert r.status_code == 200
    res = r.json()["verification"]["result"]
    assert res["verdict"] == "Inconclusive" and res["providers"][0]["status"] == "error"


def test_verify_works_when_model_is_missing(make_client):
    r = make_client(providers=[StubProvider()]).post("/api/verify", json={"text": CLAIM_TEXT})
    body = r.json()
    assert r.status_code == 200 and body["ml"] is None and "not available" in body["ml_note"]
    assert body["verification"]["triggered"] is True


def test_verify_validates_input(make_client):
    assert make_client(predictor=FakePredictor()).post("/api/verify", json={"text": ""}).status_code == 422


# ---------- verify-url ----------
ARTICLE_HTML = ("<html><head><title>Clinics plan</title></head><body><article><p>"
                + CLAIM_TEXT + " The scheme is expected to cost a large sum and will be reviewed "
                "by the state cabinet over the coming months as planned by the ministry.</p>"
                "<script>alert('x')</script></article></body></html>")


def mock(handler):
    return httpx.MockTransport(handler)


def test_verify_url_happy_path(make_client):
    seen = {}

    def handler(req):
        seen["host"] = req.headers["host"]
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE_HTML)
    c = make_client(predictor=FakePredictor(True), transport=mock(handler))
    r = c.post("/api/verify-url", json={"url": "https://news.example.com/a"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"]["title"] == "Clinics plan" and seen["host"] == "news.example.com"
    assert body["verification"]["triggered"]


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://localhost/", "file:///etc/passwd",
                                 "http://169.254.169.254/latest/meta-data/", "ftp://x.com/a",
                                 "http://user:pw@example.com/", "http://[::1]/"])
def test_verify_url_blocks_ssrf_attempts(make_client, url):
    r = make_client(predictor=FakePredictor()).post("/api/verify-url", json={"url": url})
    assert r.status_code == 400 and r.json()["message"].startswith("URL not allowed")


def test_verify_url_blocks_hostname_resolving_to_private(tmp_path):
    s = Settings(_env_file=None, feedback_path=str(tmp_path / "f"),
                 rate_limit_per_minute=1000, monitoring_log_path=str(tmp_path / "ev.jsonl"))

    async def private_resolver(host, port):
        return ["10.0.0.7"]
    app = create_app(s, predictor=FakePredictor(), verifier=Verifier([StubProvider()], s, SourceTiers()),
                     resolver=private_resolver)
    r = TestClient(app).post("/api/verify-url", json={"url": "https://sneaky.example.com/"})
    assert r.status_code == 400


def test_verify_url_upstream_failures_map_to_clean_errors(make_client):
    def pdf(req):
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"x")
    c = make_client(predictor=FakePredictor(), transport=mock(pdf))
    assert c.post("/api/verify-url", json={"url": "https://a.example/x"}).status_code == 415
    c = make_client(predictor=FakePredictor(), transport=mock(lambda r: httpx.Response(500)))
    assert c.post("/api/verify-url", json={"url": "https://a.example/x"}).status_code == 502
    empty = httpx.Response(200, headers={"content-type": "text/html"}, text="<html><body>hi</body></html>")
    c = make_client(predictor=FakePredictor(), transport=mock(lambda r: empty))
    assert c.post("/api/verify-url", json={"url": "https://a.example/x"}).status_code == 422


# ---------- feedback / metrics ----------
def test_feedback_stores_hash_only_by_default_and_text_only_with_consent(make_client, tmp_path):
    c = make_client(predictor=FakePredictor())
    r = c.post("/api/feedback", json={"user_label": "fake", "predicted_label": "real",
                                      "text": "secret article body"})
    assert r.status_code == 201 and r.json()["stored_text"] is False
    c.post("/api/feedback", json={"user_label": "real", "predicted_label": "real",
                                  "text": "shared body", "consent_store_text": True})
    lines = [json.loads(x) for x in (tmp_path / "fb.jsonl").read_text().splitlines()]
    assert lines[0]["text"] is None and lines[0]["text_sha256"]
    assert "secret article body" not in (tmp_path / "fb.jsonl").read_text()
    assert lines[1]["text"] == "shared body"
    fb = c.get("/api/metrics").json()["feedback"]
    assert fb["total"] == 2 and fb["user_agreement_with_model"] == {"agree": 1, "disagree": 1}
    assert "NOT live model accuracy" in fb["note"]


def test_feedback_rejects_bad_label(make_client):
    c = make_client(predictor=FakePredictor())
    assert c.post("/api/feedback", json={"user_label": "maybe"}).status_code == 422


def test_metrics_reflect_actual_traffic(make_client):
    c = make_client(predictor=FakePredictor(True),
                    providers=[StubProvider(error=ProviderUnavailable("down"))])
    c.post("/api/predict", json={"text": CLAIM_TEXT})
    c.post("/api/verify", json={"text": CLAIM_TEXT})
    c.post("/api/predict", json={"text": ""})
    m = c.get("/api/metrics").json()
    assert m["routes"]["POST /api/predict"]["count"] == 2
    assert m["routes"]["POST /api/predict"]["errors_4xx"] == 1
    assert m["predictions"]["total"] == 2 and m["model_version"] == "fake-1"
    assert m["verification"]["incomplete"] == 1
    assert m["verification_provider_failures"] == {"stub": 1}
    assert not any("accuracy" in k.lower() for k in _all_keys(m))  # no metric claims accuracy


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _all_keys(v)


# ---------- resilience ----------
def test_rate_limit_returns_429_with_retry_after(make_client):
    c = make_client(predictor=FakePredictor(), rate_limit_per_minute=3)
    codes = [c.post("/api/predict", json={"text": CLAIM_TEXT}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = c.post("/api/predict", json={"text": CLAIM_TEXT})
    assert int(r.headers["retry-after"]) >= 1
    assert c.get("/api/health").status_code == 200       # health is not rate limited


def test_unhandled_errors_do_not_leak_internals(make_client):
    class Exploding(Verifier):
        async def verify(self, text):
            raise RuntimeError("secret stack detail /home/app/key=abc")
    s = Settings(_env_file=None)
    v = Exploding([StubProvider()], s, SourceTiers())
    c = make_client(predictor=FakePredictor(True), verifier=v)
    r = c.post("/api/verify", json={"text": CLAIM_TEXT})
    assert r.status_code == 500
    assert r.json()["error"] == "internal_error"
    assert "secret" not in r.text and "Traceback" not in r.text and "/home" not in r.text


def test_unknown_route_404_is_json(make_client):
    r = make_client().get("/api/nope")
    assert r.status_code == 404 and r.json()["error"] == "http_404"


def test_verify_url_unresolvable_host_gives_friendly_message(tmp_path):
    s = Settings(_env_file=None, feedback_path=str(tmp_path / "f"),
                 rate_limit_per_minute=1000, monitoring_log_path=str(tmp_path / "ev.jsonl"))

    async def nothing(host, port):
        return []
    app = create_app(s, predictor=FakePredictor(),
                     verifier=Verifier([StubProvider()], s, SourceTiers()), resolver=nothing)
    r = TestClient(app).post("/api/verify-url", json={"url": "https://typo.example.com/"})
    assert r.status_code == 400 and "not allowed" not in r.json()["message"]
    assert "resolve" in r.json()["message"]


# ---------- monitoring endpoints ----------
def test_events_logged_without_article_text(make_client, real_predictor, tmp_path):
    c = make_client(predictor=real_predictor)
    secret = "zebra-quartz-secret-sentence " * 8
    c.post("/api/predict", json={"text": secret})
    raw = (tmp_path / "events.jsonl").read_text()
    assert "zebra" not in raw and "quartz" not in raw
    assert json.loads(raw.splitlines()[0])["model_version"] == "vtest"


def test_drift_endpoint_insufficient_then_detects_unfamiliar_traffic(make_client, real_predictor):
    c = make_client(predictor=real_predictor, drift_min_samples=10)
    assert c.get("/api/monitoring/drift").json()["status"] == "insufficient_data"
    for _ in range(12):
        c.post("/api/predict", json={"text": UNRELATED})
    rep = c.get("/api/monitoring/drift").json()
    assert rep["status"] == "drift" and rep["model_version"] == "vtest"
    assert any(f["feature"] == "novelty_score" and f["status"] == "drift" for f in rep["features"])
    assert any(r["rate"] == "ood_rate" and r["status"] == "drift" for r in rep["rates"])


def test_drift_endpoint_without_model_is_503(make_client):
    assert make_client().get("/api/monitoring/drift").status_code == 503


def test_analytics_counts_match_traffic(make_client, real_predictor):
    c = make_client(predictor=real_predictor)
    assert c.get("/api/analytics").json()["total"] == 0
    for _ in range(3):
        c.post("/api/predict", json={"text": CLAIM_TEXT})
    a = c.get("/api/analytics").json()
    assert a["total"] == 3 and sum(a["labels"].values()) == 3
    assert sum(d["total"] for d in a["by_day"]) == 3 and "not accuracy" in a["note"]


def test_performance_endpoint_requires_reviewed_labels(make_client, tmp_path):
    from ml.monitoring import review
    c = make_client(predictor=FakePredictor(), min_review_labels=2)
    out = c.get("/api/monitoring/performance").json()
    assert out["status"] == "insufficient_labels" and out["metrics"] is None
    ids = []
    for pred, text in (("real", "article one body"), ("fake", "article two body")):
        r = c.post("/api/feedback", json={"user_label": "fake", "predicted_label": pred,
                                          "text": text, "consent_store_text": True,
                                          "model_version": "fake-1"})
        ids.append(r.json()["id"])
    assert c.get("/api/monitoring/performance").json()["status"] == "insufficient_labels"
    review.set_review(tmp_path / "fb.jsonl", ids[0], "real", "alice")
    review.set_review(tmp_path / "fb.jsonl", ids[1], "real", "alice")
    out = c.get("/api/monitoring/performance").json()
    assert out["status"] == "ok" and out["metrics"]["n"] == 2
    assert out["metrics"]["accuracy"] == 0.5 and "self-selected" in out["note"]


def test_experiments_endpoint_lists_recorded_versions(make_client, real_predictor, trained_models):
    c = make_client(predictor=real_predictor, model_dir=str(trained_models[0]))
    body = c.get("/api/experiments").json()
    assert body["active"] == "vtest" and body["versions"][0]["synthetic_data"] is True
    assert "confusion_matrix" not in body["versions"][0]["test_metrics"]
    assert make_client().get("/api/experiments").json()["versions"] == []
