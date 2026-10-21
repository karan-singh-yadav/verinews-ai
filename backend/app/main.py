from __future__ import annotations

import logging
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.core.metrics import MetricsCollector
from backend.app.core.ratelimit import SlidingWindowRateLimiter
from backend.app.core.security import Resolver
from backend.app.services.evidence import load_source_tiers
from backend.app.services.feedback import FeedbackStore
from backend.app.services.search.factcheck import FactCheckProvider
from backend.app.services.search.gdelt import GdeltProvider
from backend.app.services.verifier import Verifier
from ml.inference.predictor import ModelNotAvailableError, Predictor
from ml.monitoring.events import PredictionLog

MAX_BODY_BYTES = 200_000
logger = logging.getLogger("verinews.api")


def build_verifier(s: Settings) -> Verifier:
    providers = [
        FactCheckProvider(s.google_fact_check_api_key, timeout=s.request_timeout_seconds,
                          language=s.google_fact_check_language or None),
        GdeltProvider(enabled=s.gdelt_enabled, timeout=s.request_timeout_seconds + 5,
                      min_interval=s.gdelt_min_interval_seconds),
    ]
    return Verifier(providers, s, load_source_tiers(s.sources_config))


def create_app(
    settings: Settings | None = None, predictor: Predictor | None = None,
    verifier: Verifier | None = None, resolver: Resolver | None = None,
    fetch_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    s = settings or get_settings()
    configure_logging()
    app = FastAPI(title="VeriNews AI", version="0.1.0",
                  description="Fake news classification with OOD detection and live verification")
    st = app.state
    st.settings, st.metrics = s, MetricsCollector()
    st.rate_limiter = SlidingWindowRateLimiter(s.rate_limit_per_minute)
    st.feedback = FeedbackStore(s.feedback_path)
    st.events = PredictionLog(s.monitoring_log_path, s.monitoring_enabled)
    st.verifier = verifier or build_verifier(s)
    st.resolver, st.fetch_transport = resolver, fetch_transport
    st.predictor = predictor
    if st.predictor is None:
        try:
            st.predictor = Predictor.load(s.model_dir)
        except ModelNotAvailableError:
            logger.warning("no trained model found; /api/predict will return 503 until trained")

    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origin_list,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = uuid.uuid4().hex
        request.state.request_id = rid
        started = time.perf_counter()
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            response = JSONResponse(status_code=413, content={
                "error": "payload_too_large", "message": "Request body too large.",
                "request_id": rid})
        else:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("unhandled error", extra={"request_id": rid})
                response = JSONResponse(status_code=500, content={
                    "error": "internal_error", "message": "An unexpected error occurred.",
                    "request_id": rid})
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        st.metrics.record_request(path, request.method, response.status_code, elapsed)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        logger.info("request", extra={"request_id": rid, "method": request.method, "path": path,
                                      "status": response.status_code,
                                      "ms": round(elapsed * 1000, 1)})
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, headers=getattr(exc, "headers", None),
                            content={"error": f"http_{exc.status_code}", "message": str(exc.detail),
                                     "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Report field + message only; never echo submitted input back.
        problems = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"),
                     "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "error": "validation_error", "message": "Invalid request.", "details": problems,
            "request_id": getattr(request.state, "request_id", None)})

    app.include_router(router)
    return app


app = create_app()
