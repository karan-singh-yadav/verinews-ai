# Troubleshooting (Windows first)

| Symptom | Cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell policy. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen the terminal |
| `python` opens the Microsoft Store or is not found | Install Python 3.12 from python.org (tick "Add to PATH"), or use `py -3.12`. Turn off the Store "App execution aliases" for python |
| `ImportError: cannot import name 'UTC' from 'datetime'` | Python is older than 3.11. Check `python --version`, recreate the venv with 3.11+ (3.12 is the tested version) |
| `pip` fails building `lxml` / `scikit-learn` | Very new or very old Python without prebuilt wheels. Use Python 3.12 and upgrade pip: `python -m pip install -U pip` |
| `uvicorn is not recognized` | The venv is not active, or use `python -m uvicorn backend.app.main:app --reload` |
| `ModuleNotFoundError: No module named 'ml'` / `backend` | Run commands from the repository root (the folder containing `pyproject.toml`) |
| `FileNotFoundError: ... Fake.csv not found` | Put `Fake.csv` and `True.csv` in `data\raw\`. Kaggle CLI needs `%USERPROFILE%\.kaggle\kaggle.json` |
| Training is very slow or runs out of memory | Lower `word_max_features` and `char_max_features` in `configs/training.yaml`. (I could not measure real-data training here) |
| Sidebar says "No model trained yet" / API returns 503 | Train first: `python -m ml.training.train --dataset isot --data-dir data\raw`. If the API started with no model, the next request after training loads it. If an older model was already loaded (or after `ml.registry promote`), restart the API |
| Verification says "incomplete" | No provider answered. The panel lists each provider's reason. Typical: `GOOGLE_FACT_CHECK_API_KEY is not set`, or GDELT unreachable |
| Fact Check: "rejected the request (HTTP 403)" | Key invalid, or the Fact Check Tools API is not enabled for the key's Google Cloud project, or the key's restrictions block this server |
| Fact Check: "rate limit or quota exceeded" | Wait, or check the quota shown in your Google Cloud console |
| GDELT: "rate limit hit" | GDELT asks for about one request per 5 seconds; the client already spaces calls, so retry in a few seconds. GDELT only covers roughly the last three months |
| GDELT or Fact Check unreachable from your network | Corporate proxy or firewall. Allow `api.gdeltproject.org` and `factchecktools.googleapis.com` |
| "URL not allowed" when checking a link | Intentional SSRF protection: only public http(s) pages on ports 80/443 are fetched. Local, private and non-web addresses are refused |
| "Could not extract enough article text" | Paywalled or JavaScript-rendered page. Paste the text instead |
| Browser shows "Can't reach the API" | Backend not running on port 8000, or the frontend points elsewhere (`VITE_API_BASE`). With `npm run dev` the proxy target is `VITE_PROXY_TARGET` (default `http://localhost:8000`) |
| CORS error when the frontend runs on another origin | Add the origin to `CORS_ORIGINS` in `.env` and restart the API |
| `429 Too many requests` | Rate limit (30/min per client by default, `RATE_LIMIT_PER_MINUTE`). Behind another proxy every visitor may share one bucket: see `docs/DEPLOYMENT.md` |
| Port already in use | `netstat -ano \| findstr :8000` shows the PID; stop it or pick another port (`--port 8001`, `WEB_PORT=8081`) |
| `npm install` fails or odd Vite errors | Use Node 22 (`node --version`). Delete `frontend\node_modules` and run `npm ci`. Antivirus can cause `EPERM`: exclude the project folder |
| Drift page says "Not enough data yet" | Needs `DRIFT_MIN_SAMPLES` (100) logged requests. "No reference profile" means the model was trained before monitoring existed: retrain |
| Quality page says "No accuracy figure yet" | By design: needs 50 expert-reviewed examples. Review with `python -m ml.monitoring.review --list` |
| `StarletteDeprecationWarning: Using httpx with starlette.testclient` | Harmless warning from the test client; tests still pass |
| `mlflow ui` shows nothing | Run it from the repo root with `--backend-store-uri sqlite:///mlflow.db`; the database appears after the first training run |
| Docker: `env_file ... required` error | Needs Compose v2.24+. Update Docker Desktop, or `copy .env.example .env` and remove the `required: false` line |
| Docker on Linux: `Permission denied` writing `models/` | The container runs as UID 10001: `sudo chown -R 10001 models` |
| Shell scripts fail with `bad interpreter` or `\r` | CRLF line endings. `.gitattributes` forces LF; re-clone or run `git add --renormalize .` |
| OneDrive folder, path-length or locking errors | Move the project to a short local path such as `C:\dev\verinews-ai` |
