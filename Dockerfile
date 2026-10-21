# syntax=docker/dockerfile:1
# Backend image (API + training + MLflow server all run from this one image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Exact, test-verified runtime versions (see requirements.lock.txt).
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY backend ./backend
COPY ml ./ml
COPY configs ./configs
COPY scripts ./scripts

# Run as an unprivileged user; /app/models, /app/data and /app/mlruns are meant to be volumes.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/models /app/data /app/mlruns \
    && chown -R app:app /app
USER 10001

ENV MODEL_DIR=/app/models \
    MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db \
    FEEDBACK_PATH=/app/data/feedback/feedback.jsonl \
    MONITORING_LOG_PATH=/app/data/monitoring/events.jsonl \
    FORWARDED_ALLOW_IPS=127.0.0.1

EXPOSE 8000

# /api/health answers 200 even without a trained model ("degraded"), so the container stays up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

# --proxy-headers makes the rate limiter see the real client IP (from nginx) instead of the proxy's.
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips \"$FORWARDED_ALLOW_IPS\""]
