"""End-to-end smoke test against a real running server. Works on Windows, macOS and Linux.

  python scripts/smoke_test.py --self-contained     # trains a SYNTHETIC model, starts uvicorn, checks it
  python scripts/smoke_test.py --against http://localhost:8080   # checks an already running stack

--self-contained uses synthetic data only to exercise the plumbing; the metrics it produces mean nothing.
Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def run_checks(base: str, expect_model: bool) -> None:
    c = httpx.Client(base_url=base, timeout=90)
    h = c.get("/api/health")
    check("health responds 200", h.status_code == 200, h.text[:120])
    health = h.json()
    check("health reports model state", health["model_loaded"] is expect_model, str(health))
    check("security headers present", h.headers.get("x-content-type-options") == "nosniff")
    check("request id header present", bool(h.headers.get("x-request-id")))

    bad = c.post("/api/predict", json={"text": "   "})
    check("empty text rejected with 422", bad.status_code == 422)
    check("no stack trace in errors", "Traceback" not in bad.text)

    for url in ("http://169.254.169.254/latest/meta-data/", "http://localhost/", "file:///etc/passwd",
                "http://[::1]/", "http://user:pw@example.com/"):
        r = c.post("/api/verify-url", json={"url": url})
        check(f"SSRF refused: {url}", r.status_code == 400, f"got {r.status_code}")

    if not expect_model:
        r = c.post("/api/predict", json={"text": "some article text " * 10})
        check("predict without a model is 503", r.status_code == 503)
        return

    text = ("The health ministry announced that 4,500 new clinics will open in Uttar Pradesh next "
            "year, according to officials in Lucknow.")
    p = c.post("/api/predict", json={"text": text})
    check("predict returns 200", p.status_code == 200, p.text[:150])
    pred = p.json()["prediction"]
    check("prediction fields valid", pred["label"] in ("real", "fake") and 0 <= pred["fake_probability"] <= 1)

    v = c.post("/api/verify", json={"text": text, "force_verification": True})
    check("verify returns 200", v.status_code == 200, v.text[:150])
    res = v.json()["verification"]["result"]
    check("verification result present", res is not None)
    check("verdict is one of the four allowed",
          res["verdict"] in ("Supported", "Contradicted", "Mixed Evidence", "Inconclusive"))
    if not res["completed"]:
        check("incomplete verification is Inconclusive (never a guess)", res["verdict"] == "Inconclusive")
    print(f"        verification completed={res['completed']} verdict={res['verdict']}")
    print("        providers: " + "; ".join(f"{x['name']}={x['status']}" for x in res["providers"]))

    check("feedback accepted", c.post("/api/feedback", json={"user_label": "unsure"}).status_code == 201)
    for path in ("/api/model-info", "/api/experiments", "/api/analytics", "/api/metrics",
                 "/api/monitoring/drift", "/api/monitoring/performance"):
        check(f"GET {path} is 200", c.get(path).status_code == 200)
    a = c.get("/api/analytics").json()
    check("analytics counted our requests", a["total"] >= 1, str(a.get("total")))


def start_server(models: Path, tmp: Path, port: int) -> subprocess.Popen:
    env = {**os.environ, "MODEL_DIR": str(models), "FEEDBACK_PATH": str(tmp / "fb.jsonl"),
           "MONITORING_LOG_PATH": str(tmp / "events.jsonl"), "GDELT_MIN_INTERVAL_SECONDS": "0",
           "RATE_LIMIT_PER_MINUTE": "1000", "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(port),
         "--log-level", "warning", "--proxy-headers"], cwd=ROOT, env=env)


def wait_ready(base: str, seconds: int = 40) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            if httpx.get(f"{base}/api/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-contained", action="store_true")
    ap.add_argument("--against")
    ap.add_argument("--no-model", action="store_true", help="with --against: expect no trained model")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not a.self_contained and not a.against:
        ap.error("choose --self-contained or --against URL")

    if a.against:
        print(f"Checking {a.against}")
        run_checks(a.against.rstrip("/"), expect_model=not a.no_model)
    else:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            print("Training a SYNTHETIC model (plumbing test only)...")
            subprocess.run([sys.executable, "scripts/make_synthetic_fixture.py", "--out",
                            str(tmp / "data"), "--n-per-class", "300"], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, "-m", "ml.training.train", "--data-dir", str(tmp / "data"),
                            "--output-dir", str(tmp / "models"), "--version", "smoke", "--no-mlflow",
                            "--synthetic"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": str(ROOT)})
            base = f"http://127.0.0.1:{a.port}"
            proc = start_server(tmp / "models", tmp, a.port)
            try:
                if not wait_ready(base):
                    print("Server did not become ready")
                    return 1
                run_checks(base, expect_model=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
    print(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
