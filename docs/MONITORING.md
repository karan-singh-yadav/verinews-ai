# Monitoring, drift and real-world performance

## What is logged
Every `/api/predict` and `/api/verify*` call appends one line to `data/monitoring/events.jsonl`:
model version, label, probabilities, novelty score, word/char counts, OOD / low-confidence flags,
verification verdict. **Never** article text, URLs or IP addresses (`MONITORING_ENABLED=false`
turns it off). A logging failure never breaks a request.

## Drift (inputs / outputs changed vs training time)
* Reference = the **test split** at training time (`models/<v>/reference_features.csv` +
  `metadata.json -> monitoring_reference`). It was never used for fitting or thresholds.
* Per feature: PSI (effect size) + KS test (significance). `drift` needs PSI >= 0.25 **and**
  KS p < 0.01; `warning` needs PSI >= 0.10 **and** KS p < 0.05. Rates (fake / out-of-distribution
  / low-confidence share) use a two-proportion z-test plus a 5-point minimum difference.
* Below `DRIFT_MIN_SAMPLES` (100) requests the answer is `insufficient_data`: **no drift claim**.
* Bins adapt to sample size. A reference under 300 rows is flagged `reference_small`.
* Measured on simulated same-distribution data (200 trials, live n=300): false-alarm rate was
  97% (reference n=90) before the adaptive-bin + KS-gate fix and is now 8% (n=90), 2% (n=500),
  2% (n=3000). Strong shifts were detected in 100/100 runs. Small shifts were not characterised.
* **Drift is not accuracy.** It says "the model sees different data", nothing more.

Run offline: `python -m ml.monitoring.report [--evidently]` -> `reports/drift.json`
(+ `drift_evidently.html`, needs `pip install -r requirements-monitoring.txt`).
Live: `GET /api/monitoring/drift`.

## Real-world performance (the only honest way)
Live accuracy is **never** estimated from confidence, user votes or verification verdicts.
1. Users may submit feedback (`POST /api/feedback`). Text is stored only with explicit consent.
2. A reviewer labels records using primary sources / fact-checks:
   `python -m ml.monitoring.review --list`  then
   `python -m ml.monitoring.review --label <id> fake --reviewer <name>`
3. `GET /api/monitoring/performance` reports accuracy (Wilson 95% CI), precision, recall, F1 and
   a confusion matrix **only after >= `MIN_REVIEW_LABELS` (50) reviewed examples**; below that it
   returns `insufficient_labels` and no numbers.
4. Feedback is self-selected, so this is biased. For an unbiased estimate, review a **random
   sample** of requests (this needs a deployment policy that retains sampled text with consent).

## Model versions
`python -m ml.registry list` / `promote <version>` (rollback = promote an older version, then
restart the API). Each version keeps its own metadata, metrics and reference profile.
