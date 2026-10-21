from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from backend.app.core.security import (
    FetchError,
    HostResolutionError,
    UnsafeURLError,
    UnsupportedContentError,
    fetch_url,
)
from backend.app.schemas.api import (
    DISCLAIMER,
    MAX_TEXT_CHARS,
    FeedbackRequest,
    PredictRequest,
    VerifyRequest,
    VerifyUrlRequest,
)
from backend.app.services.article_extract import ExtractionError, extract_article
from ml.inference.predictor import InvalidInputError, ModelNotAvailableError, Predictor
from ml.monitoring.drift import DriftConfig, build_report
from ml.monitoring.performance import evaluate_reviewed
from ml.registry import active_version, list_versions

router = APIRouter(prefix="/api")


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _rate_limit(request: Request) -> None:
    ok, retry = request.app.state.rate_limiter.check(
        request.client.host if request.client else "unknown")
    if not ok:
        raise HTTPException(429, "Rate limit exceeded. Please slow down.",
                            headers={"Retry-After": str(retry)})


def _predictor(request: Request) -> Predictor:
    """Lazy-load so a model trained after startup is picked up without restarting."""
    state = request.app.state
    if state.predictor is None:
        try:
            state.predictor = Predictor.load(state.settings.model_dir)
        except ModelNotAvailableError as exc:
            raise HTTPException(503, f"No trained model is available. {exc}") from exc
    return state.predictor


async def _classify(request: Request, text: str):
    predictor = _predictor(request)
    try:
        pred = await run_in_threadpool(predictor.predict, text)
    except InvalidInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    request.app.state.metrics.record_prediction(pred)
    return pred


async def _verify_flow(request: Request, text: str, force: bool) -> dict:
    state = request.app.state
    pred, ml_note = None, None
    try:
        pred = await _classify(request, text)
    except HTTPException as exc:
        if exc.status_code != 503:  # model missing is tolerated; invalid input is not
            raise
        ml_note = "The ML model is not available; online verification was run without it."

    triggered = force or pred is None or pred.needs_verification
    reasons: list[str] = []
    if pred is not None and pred.needs_verification:
        reasons += pred.verification_reasons
    if force:
        reasons.append("Online verification was requested by the user.")
    if pred is None:
        reasons.append("No ML prediction was available to judge confidence.")

    verification: dict = {"triggered": bool(triggered), "reasons": reasons, "result": None}
    if triggered:
        result = await state.verifier.verify(text)
        state.metrics.record_verification(result)
        verification["result"] = result
    else:
        verification["note"] = (
            "The model was confident and the text looks familiar, so online verification was "
            "not triggered. Model confidence is not proof; use force_verification to check anyway.")
    if pred is not None:
        state.events.record(pred, verification)
    return {"request_id": _rid(request), "ml": pred.to_dict() if pred else None,
            "ml_note": ml_note, "verification": verification, "disclaimer": DISCLAIMER}


@router.post("/predict")
async def predict(request: Request, body: PredictRequest):
    _rate_limit(request)
    pred = await _classify(request, body.text)
    request.app.state.events.record(pred)
    return {"request_id": _rid(request), "prediction": pred.to_dict(), "disclaimer": DISCLAIMER}


@router.post("/verify")
async def verify(request: Request, body: VerifyRequest):
    _rate_limit(request)
    return await _verify_flow(request, body.text, body.force_verification)


@router.post("/verify-url")
async def verify_url(request: Request, body: VerifyUrlRequest):
    _rate_limit(request)
    s = request.app.state.settings
    try:
        page = await fetch_url(
            body.url, timeout=s.request_timeout_seconds, max_bytes=s.max_fetch_bytes,
            max_redirects=s.max_redirects,
            resolver=request.app.state.resolver, transport=request.app.state.fetch_transport)
    except HostResolutionError as exc:
        raise HTTPException(400, str(exc)) from exc
    except UnsafeURLError as exc:
        raise HTTPException(400, f"URL not allowed: {exc}") from exc
    except UnsupportedContentError as exc:
        raise HTTPException(415, str(exc)) from exc
    except FetchError as exc:
        raise HTTPException(502, str(exc)) from exc
    try:
        article = extract_article(page.text, max_chars=min(s.max_article_chars, MAX_TEXT_CHARS))
    except ExtractionError as exc:
        raise HTTPException(422, str(exc)) from exc

    text = f"{article.title}. {article.text}" if article.title else article.text
    out = await _verify_flow(request, text[:MAX_TEXT_CHARS], body.force_verification)
    out["source"] = {"url": page.final_url, "title": article.title, "published": article.published,
                     "redirects": page.redirects,
                     "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
    return out


@router.get("/health")
async def health(request: Request):
    state = request.app.state
    try:
        pred = _predictor(request)
        model_loaded, version = True, pred.version
    except HTTPException:
        model_loaded, version = False, None
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded,
            "model_version": version, "providers_configured": state.verifier.provider_overview(),
            "time": datetime.now(UTC).isoformat(timespec="seconds")}


@router.get("/model-info")
async def model_info(request: Request):
    m = _predictor(request).metadata
    return {
        "model_version": m["model_version"], "selected_model": m["selected_model"],
        "trained_at": m["trained_at"], "synthetic_data": m.get("synthetic_data", False),
        "dataset": {k: m["dataset"].get(k) for k in
                    ("type", "rows_raw", "rows_after_dedup", "class_counts", "split_sizes")},
        "validation_comparison": m["validation_comparison"], "test_metrics": m["test_metrics"],
        "calibration": m["calibration"], "confidence_threshold": m["confidence_threshold"],
        "novelty": m["novelty"], "limitations": m["limitations"],
    }


@router.get("/metrics")
async def metrics(request: Request):
    state = request.app.state
    return {**state.metrics.snapshot(), "feedback": state.feedback.summary()}


@router.post("/feedback", status_code=201)
async def feedback(request: Request, body: FeedbackRequest):
    _rate_limit(request)
    fid = request.app.state.feedback.add(
        user_label=body.user_label, predicted_label=body.predicted_label,
        model_version=body.model_version, request_id=body.request_id, comment=body.comment,
        text=body.text, consent_store_text=body.consent_store_text)
    return {"id": fid, "stored_text": bool(body.consent_store_text and body.text),
            "message": "Thank you. Feedback is stored for review and is not used to "
                       "retrain the model automatically."}


# ---------------- monitoring ----------------
@router.get("/monitoring/drift")
async def monitoring_drift(request: Request):
    state, s = request.app.state, request.app.state.settings
    predictor = _predictor(request)

    def compute() -> dict:
        current = state.events.load(s.drift_window, predictor.version)
        return build_report(predictor.reference_features(),
                            predictor.metadata.get("monitoring_reference"), current,
                            DriftConfig(min_samples=s.drift_min_samples))
    report = await run_in_threadpool(compute)
    return {**report, "model_version": predictor.version}


@router.get("/monitoring/performance")
async def monitoring_performance(
    request: Request, model_version: str | None = Query(None, max_length=64)
):
    s = request.app.state.settings
    return await run_in_threadpool(
        evaluate_reviewed, s.feedback_path, s.min_review_labels, model_version)


@router.get("/analytics")
async def analytics(request: Request):
    """Aggregates over logged events (derived numbers only)."""
    state = request.app.state
    df = await run_in_threadpool(state.events.load, 10000)
    if df.empty:
        return {"total": 0, "by_day": [], "labels": {}, "verdicts": {}, "rates": {},
                "note": "No requests have been logged yet."}
    df = df.assign(day=df["ts"].str[:10])
    by_day = (df.groupby("day").agg(
        total=("label", "size"), fake=("label", lambda x: int((x == "fake").sum())),
        ood=("ood", lambda x: int(x.astype(bool).sum())),
        verifications=("verification_triggered", lambda x: int(x.astype(bool).sum())))
        .reset_index().tail(30))
    return {
        "total": int(len(df)),
        "by_day": by_day.astype({"total": int}).to_dict(orient="records"),
        "labels": {k: int(v) for k, v in df["label"].value_counts().items()},
        "verdicts": {k: int(v) for k, v in df["verdict"].dropna().value_counts().items()},
        "rates": {"out_of_distribution": float(df["ood"].astype(bool).mean()),
                  "low_confidence": float(df["low_confidence"].astype(bool).mean()),
                  "verification_triggered":
                      float(df["verification_triggered"].astype(bool).mean())},
        "note": "Counts of logged requests only; not accuracy.",
    }


@router.get("/experiments")
async def experiments(request: Request):
    root = request.app.state.settings.model_dir
    versions = await run_in_threadpool(list_versions, root)
    return {"active": active_version(root), "versions": versions,
            "note": "Recorded training results per model version. Open the MLflow UI "
                    "(mlflow ui --backend-store-uri sqlite:///mlflow.db) for full run history."}
