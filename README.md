# VeriNews AI: fake news detection with live verification

A portfolio project that combines **machine learning**, **NLP**, **online research** and **MLOps**:

1. **Fake-news classifier**: TF-IDF (word + character n-grams) with Logistic Regression / Linear SVM,
   calibrated probabilities, and leakage-safe training.
2. **Unfamiliar-news detection**: confidence and out-of-distribution checks that decide *when the model
   should not be trusted*.
3. **Live verification**: for uncertain or unfamiliar text, search real fact-check and news sources and
   return **Supported / Contradicted / Mixed Evidence / Inconclusive**, with citations.
4. **MLOps**: MLflow, DVC, model registry, drift monitoring, Docker, GitHub Actions, tests.

> **The ML score is not proof of anything.** It reads writing style learned from one dataset. Online evidence is
> assessed separately, and "no evidence found" is reported as *Inconclusive*, never as "fake".

## Screenshots
<img width="1352" height="629" alt="Screenshot (17)" src="https://github.com/user-attachments/assets/2578f241-341a-4153-8178-21deba448980" />
<img width="1353" height="635" alt="Screenshot (18)" src="https://github.com/user-attachments/assets/1efeda0e-4aec-47e4-899a-d6c7e11917c8" />
<img width="1351" height="636" alt="Screenshot (19)" src="https://github.com/user-attachments/assets/9c1ca421-d105-4fd7-87fd-d74d03cdc394" />
<img width="1350" height="636" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/a2a1efee-b2bd-4591-a07d-ecfc05579660" />
<img width="1366" height="626" alt="Screenshot (21)" src="https://github.com/user-attachments/assets/962ee8ee-f18f-4303-9aac-38613bb97cec" />
<img width="1366" height="624" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/1c9c37e0-eae0-4f06-b737-6488842e0d48" />
<img width="1366" height="640" alt="Screenshot (23)" src="https://github.com/user-attachments/assets/34483166-f0fd-4210-89ad-e1bc737f8e10" />
<img width="1366" height="639" alt="Screenshot (24)" src="https://github.com/user-attachments/assets/038d49b8-feed-477d-acdf-f1430eaf85af" />
<img width="1366" height="639" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/42698f94-0638-43ba-bfc0-1116c5a1a474" />
<img width="1366" height="640" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/ddb4f0a3-13bb-4cfc-a7d6-1c2283d1f175" />


## Verification status (what was actually run)

| Verified in the build environment | Result |
|---|---|
| Backend tests (`pytest`), including in a clean virtualenv with newer dependency versions | 190 pass |
| Frontend type-check, 32 component/flow tests, production build | pass |
| Frontend against the real backend (`vitest`, live) | 3 pass |
| `scripts/smoke_test.py` against a real uvicorn server | 25 checks pass |
| Real nginx in front of real uvicorn: static files, SPA fallback, security headers, proxy, per-client rate limit | 26 checks pass |
| `ruff`, `actionlint`, `hadolint`, compose YAML parsing | clean |
| DVC `repro` / skip-when-unchanged / metrics, Evidently report generation | worked (synthetic data) |

| **Not** verified | Why |
|---|---|
| Model accuracy on the real dataset | The dataset could not be downloaded here. **No real metrics exist yet**; the only trained model used synthetic data and is flagged as such |
| Live Google Fact Check / GDELT responses | Not reachable from the build environment; clients are tested against mocked HTTP |
| `docker build` / `docker compose up`, GitHub Actions runs | Docker and GitHub were unavailable |
| How the UI looks in a browser | No browser was available; layout and charts were never visually inspected |
| Windows PowerShell commands, Python 3.11 | Written for Windows but executed on Linux with Python 3.12 |

## Quick start (Windows, PowerShell)

Prerequisites: Python 3.11+ (3.12 tested), Node 22, Git. Docker Desktop is optional.

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # blocked? Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt
pytest                                 # expect all green
copy .env.example .env                 # then edit .env (see "Search providers")

# 2. Dataset: "Fake and Real News Dataset" on Kaggle (Clement Bisaillon). Put Fake.csv and True.csv in data\raw\
#    Required columns: title, text. (subject/date are ignored on purpose: they leak the label.)
#    With the Kaggle CLI configured:
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data\raw --unzip

# 3. Train (prints real test metrics when it finishes)
python -m ml.training.train --dataset isot --data-dir data\raw
mlflow ui --backend-store-uri sqlite:///mlflow.db        # optional: compare runs at http://localhost:5000

# 4. Run the API (terminal 1)
uvicorn backend.app.main:app --reload --port 8000        # docs at http://localhost:8000/docs

# 5. Run the dashboard (terminal 2)
cd frontend
npm install
npm run dev                                              # http://localhost:5173
```
Check the whole thing in one go: `python scripts\smoke_test.py --self-contained` (uses a synthetic model purely
to exercise the plumbing).

**Dataset notes.** The popular ISOT data leaks its label (real articles start with "CITY (Reuters) -"); the
preprocessing strips such artifacts, removes duplicates and conflicting labels *before* splitting, and asserts no
article appears in two splits. Scores on this dataset are often near-perfect even so; that says little about new
topics or outlets, which is why the app also checks for unfamiliar text and verifies online.

## Search providers
| Provider | Setup | Notes |
|---|---|---|
| Google Fact Check Tools API | Create an API key in a Google Cloud project, enable the *Fact Check Tools API* for it, put the key in `GOOGLE_FACT_CHECK_API_KEY` | Published fact-check reviews; many claims have none |
| GDELT DOC 2.0 | none | News metadata for roughly the last three months; about one request per 5 s |

Google's Custom Search JSON API is deliberately not used: its documentation says it is closed to new customers
and retires on 1 January 2027. Add another provider by subclassing `SearchProvider`
(`backend/app/services/search/base.py`). Check each provider's current terms and quotas yourself.
With no usable provider the API returns **Inconclusive** with `completed: false` and the reason per provider.

## Docker
```powershell
copy .env.example .env
docker compose up -d --build                       # http://localhost:8080
docker compose --profile train run --rm trainer    # train inside Docker (needs data\raw\*.csv)
python scripts\smoke_test.py --against http://localhost:8080
```
Details, HTTPS, rollback and the proxy/rate-limit caveat: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## API
| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | ML classification only |
| `POST /api/verify` | Classification + online verification when uncertain, unfamiliar, short, or forced |
| `POST /api/verify-url` | Same, for a link fetched through the SSRF-hardened fetcher |
| `GET /api/health` · `/model-info` · `/metrics` | Health (incl. provider setup), recorded training metadata, operational metrics |
| `POST /api/feedback` | Optional user feedback (text stored only with explicit consent) |
| `GET /api/monitoring/drift` · `/monitoring/performance` · `/analytics` · `/experiments` | Drift, reviewed-label quality, usage analytics, model versions |

## MLOps at a glance
* **MLflow**: parameters, validation and test metrics, artifacts per run (`sqlite:///mlflow.db`).
* **Model registry**: `python -m ml.registry list|promote <version>` (rollback = promote an older version).
* **DVC**: reproducible training stage, see [docs/DVC.md](docs/DVC.md).
* **Monitoring**: request/latency/error metrics, input and prediction drift against the training-time
  reference, and real performance **only** from expert-reviewed labels: see [docs/MONITORING.md](docs/MONITORING.md).
  Live accuracy is never claimed without verified labels.
* **CI**: GitHub Actions run lint, backend tests (Python 3.11 and 3.12), a real-server smoke test, frontend
  type-check/tests/build, and Docker build plus a stack check through nginx. A tag push publishes images to GHCR
  (it does not deploy).

## Security and privacy
SSRF-hardened URL fetching (public IPs only, redirects re-validated, connection pinned, size/time limits) ·
input validation and size caps · per-client rate limiting · no stack traces or secrets in responses ·
API keys only from the environment, never logged · retrieved content treated as untrusted data ·
article text is never logged; monitoring events hold derived numbers only · feedback stores a hash unless the
user opts in · browser history keeps no text by default · CSP and security headers on the frontend.

## Known limitations
* The classifier learns one dataset's style; accuracy there does not transfer to new topics or outlets.
* Novelty detection (LSA distance) flags unfamiliar vocabulary and topics, not unfamiliar *claims*.
* Claim and entity extraction are rule-based; they cannot tell people from organisations or places.
* News headlines are only a weak signal; stance for news is a keyword heuristic and says so in the UI.
* Source tiers (`configs/sources.yaml`) are an editable editorial list, not ground truth.
* Rate limiting, metrics and the event log are per process.

## Repository layout
```
backend/app/{api,core,schemas,services}   FastAPI app, SSRF-safe fetch, search providers, evidence assessment
ml/{training,inference,monitoring}        training, Predictor/NoveltyDetector, drift + review tools, registry
frontend/                                 React + TypeScript + Tailwind + Recharts dashboard
tests/                                    190 backend tests      scripts/   smoke test, synthetic fixture
configs/                                  training.yaml, sources.yaml
docs/                                     ARCHITECTURE, DEPLOYMENT, MONITORING, DVC, TROUBLESHOOTING
Dockerfile · docker-compose*.yml · dvc.yaml · .github/workflows/
```
More: [Architecture](docs/ARCHITECTURE.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Troubleshooting](docs/TROUBLESHOOTING.md)
