# VeriNews frontend

React 19 + TypeScript + Tailwind CSS 4 + Recharts, built with Vite.

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api to http://localhost:8000)
npm run build      # type-checks, then builds to dist/
npm test           # unit + component tests (no backend needed)
```

Start the backend first (`uvicorn backend.app.main:app --port 8000` from the repo root).

**Another API address?** Set `VITE_API_BASE` (for example `VITE_API_BASE=https://api.example.com`)
before building. Leave it empty to use same-origin `/api` behind a reverse proxy.

**Test against the real backend** (skipped by default):
```powershell
$env:VITE_LIVE_API="http://localhost:8000"; npx vitest run src/test/live.test.tsx
```

## Pages
Check news · History (stored only in this browser) · Analytics · Model (metrics, confusion matrix) ·
Experiments (model versions) · Drift and quality (drift + reviewed-label performance).

## Design rules worth keeping
* Indigo is the model's colour. Verdict colours (teal, coral, amber, slate) belong only to the online
  evidence, so a style score is never mistaken for a fact-check result.
* Orange marks monitoring alerts. Every number comes from the API; empty states are shown instead of
  sample data.
* History keeps no article text unless the user opts in (80 characters), and only the hostname of links.
