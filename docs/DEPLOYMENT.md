# Deployment

## Verified vs. not verified
Verified in the build environment: nginx config (real nginx + real uvicorn: static files, SPA fallback,
security headers, `/api` proxying, per-client rate limiting), Dockerfile lint (hadolint), workflow lint
(actionlint), compose YAML parsing. **Not verified:** actual `docker build` / `docker compose up` (Docker was
not available), GitHub Actions runs, anything on a real server.

## Option 1: Docker Compose on one machine
Prerequisites: Docker Desktop (Windows: WSL 2 backend) or Docker Engine with Compose v2.24+.

```powershell
copy .env.example .env             # add GOOGLE_FACT_CHECK_API_KEY
docker compose up -d --build       # site at http://localhost:8080
docker compose ps                  # both services should become "healthy"

# Train (needs data\raw\Fake.csv and True.csv). If the API started with no model it loads this one on the
# next request; if a model was already loaded, run: docker compose restart backend
docker compose --profile train run --rm trainer

# Optional experiment UI at http://localhost:5000 (bound to localhost: it has no login)
docker compose --profile mlflow up -d mlflow

# Check the running stack end to end
python scripts/smoke_test.py --against http://localhost:8080          # add --no-model before training
```
Development with hot reload: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build`
(site on :5173, API on :8000).

Volumes: `./models` (model versions, bind mount), `appdata` (feedback + monitoring events), `mlruns`
(MLflow database). Back these up; the images themselves are disposable.

Roll back a model: `docker compose exec backend python -m ml.registry promote <old-version>` then
`docker compose restart backend`.

## Putting it on the internet
1. **Terminate TLS in front of the stack.** The nginx container speaks plain HTTP.
2. **Keep the rate limiter honest.** The API limits by client IP. nginx forwards the connecting address
   as `X-Forwarded-For` (overwriting any client-supplied value). If *another* proxy (Caddy, a cloud load
   balancer) sits in front of this nginx, nginx would see only that proxy's address and every visitor would
   share one bucket. Tell nginx to trust it and read the real address, in `frontend/nginx.conf`:
   ```nginx
   set_real_ip_from 10.0.0.0/8;          # your proxy's address range
   real_ip_header X-Forwarded-For;
   real_ip_recursive on;
   ```
   (Not tested here. Only list networks you control.)
3. **Never publish port 8000.** The compose file sets `FORWARDED_ALLOW_IPS=*` because the API is only
   reachable from the nginx container. If you expose uvicorn directly, set it to your proxy's IP instead,
   or clients can spoof their address.
4. **Secrets** stay in `.env` (git-ignored) or your platform's secret store. Never bake them into images.
5. **MLflow** has no authentication. Keep it on localhost or behind an authenticating proxy.
6. **Scaling:** rate limiting, metrics and the event log are per process/volume. Several replicas need a
   shared store (for example Redis for the limiter) first.
7. **Memory for training:** TF-IDF with word and character n-grams on the full dataset is memory-hungry.
   I have not measured it on the real dataset; start with 8 GB and lower `word_max_features` /
   `char_max_features` in `configs/training.yaml` if needed.

## Publishing images
Push a tag (`git tag v0.1.0; git push origin v0.1.0`). `release-images.yml` builds both images and pushes
them to GitHub Container Registry. It does not deploy anywhere; pulling and running them on your host is
a manual step (`docker compose pull` after pointing `image:` at the GHCR names).
