# Production Audit — 2026-09-15

Scope: full repository inspection (backend, frontend, scripts, reports, data artifacts,
Docker, tests) before any fix. Every claim below was verified by executing code or
reading the committed artifacts — nothing is assumed.

---

## 1. Current architecture (verified)

```
Customer message
  → Flask API (backend/app/api/app.py)          request ID, structured errors, rate limit
  → SupportAgent (backend/app/services/agent.py)
      → intent classification  (TF-IDF + LogisticRegression, models/classifier_AmazonHelp.joblib)
      → retrieval              (TF-IDF cosine over resolved cases, temporal index)
      → evidence scoring       (calibrated weights, models/evidence_weights.json)
      → generation             (backend/app/generation/generator.py → provider)
      → grounding validation   (claim-vs-evidence word overlap, independent of the LLM)
      → escalation policy      (backend/app/escalation/policy.py, named reason codes)
  → JSON decision trail → static JS dashboard (frontend/, no build step)
```

Provider flow: `LLM_PROVIDER` env → `get_llm_provider()` (factory) → `GroqProvider` /
`GeminiProvider` / `MockLLMProvider`. All outputs carry `is_mock`. Generation failures
are converted to `GENERATION_FAILED` → ESCALATE (fail-safe works as designed).

Evaluation flow: scripts in `scripts/` write JSON/MD reports to `reports/`; the API
serves them read-only; the frontend renders them with no hardcoded metrics.

## 2. Discovered bugs and root causes

### BUG 1 (CRITICAL) — Golden accuracy collapsed 53.5% → 19.5% (stale artifact + remapped labels)

Evidence chain (all measured in this environment):

- `reports/baseline_results.json` (regenerated 2026-09-14): silver test accuracy
  **93.57%**, **11 intents** (no `account_access_issue` in test support).
- `reports/golden_set_evaluation.json` (regenerated 2026-09-14): golden accuracy
  **19.5%** (39/200), gap "0.7407".
- `data/golden/golden_set_with_predictions.json` + every document written before
  2026-09-14 (README, `reports/final_report.md`, `FINAL_VERIFICATION.md`): golden
  accuracy **53.5%** (107/200), **12 intents** everywhere, gap "40.3 points".
- The golden CSV and the predictions export are byte-consistent (200/200 identical
  messages and labels; provenance counts 86/107/7 match the verified state restored
  by `scripts/restore_golden_csv_from_review.py`). **The golden labels did not change.**
- The committed `models/classifier_AmazonHelp.joblib` (2026-09-14 22:27) and a fresh
  retrain from the current `data/processed/labeled_AmazonHelp.jsonl` (2026-09-12 23:33)
  produce **identical predictions and the same 19.5%** on golden.

Root cause (isolated):

1. `configs/intents.yaml` was edited 2026-09-12 23:32 (one minute before the dataset
   rebuild): the cluster→intent mapping was re-derived for a re-run of clustering
   (k=15). The YAML header itself documents the remap ("RE-MAPPED 2026-09-12") and
   cites decision-log "#19", which does not exist in `docs/decision-log.md` (18 entries).
2. `scripts/build_labeled_dataset.py` was re-run at 23:33, rewriting the silver labels
   of the whole dataset (same conversations, same cluster assignments file from 21:03).
3. The current silver labels disagree with the human-verified golden labels on
   **157/200 (78.5%)** of the same messages. By label provenance:
   - manual-review labels: silver agrees 10/86 (**11.6%**)
   - classifier-fallback labels: 27/107 (25.2%)
   - keyword-rule labels: 6/7 (85.7%)
   When the silver label is wrong, the most frequent wrong labels are
   `delivery_not_received` (59) and `delivery_delay` (32) — the split of the
   "delivered-but-missing vs wrong-item" cluster (4) changed which messages get which
   label, and a large share of previously-real-intent traffic now carries
   `general_other` (47.8% of the current test split) or delivery labels.
4. The classifier trained on those labels cannot learn what the human labeler intended
   (e.g. only 6 `account_access_issue` training examples; the intent has all but
   vanished from the silver taxonomy in practice), so it scores 19.5% against the
   unchanged human golden labels. Silver accuracy looks fine (93.6%) because it is
   graded by the same remapped labels — the circularity the project itself documented,
   now made worse.
5. `scripts/evaluate_against_golden.py` was re-run on 2026-09-14 and overwrote
   `reports/golden_set_evaluation.json` (19.5%) and `reports/baseline_results.json`
   with numbers from the remapped dataset, while all narrative documents still
   describe the verified 53.5% state. The committed model artifacts were also
   retrained (22:27) from the remapped dataset.

Fix: treat the verified state as authoritative — retrain/re-evaluate from the
pre-remap label mapping, regenerate every affected report from one consistent
dataset, and record decision #19 honestly (see `reports/golden_gap_analysis.md`).

### BUG 2 (HIGH) — Failure Analysis page rendered `[object HTMLSpanElement]`

Root cause: `frontend/js/pages/failures.js` called
`card(el("span", ...), ...)` — `UI.card(title, ...)` sets `el("h2", { text: title })`,
and `textContent = <DOM element>` coerces the element to the string
`"[object HTMLSpanElement]"`. Data was fine; the title element was a DOM node.

Fix: pass a plain string as the card title and render the count badge inside the
header actions slot instead.

### BUG 3 (HIGH) — `LIVE — GROQ` badge while every generation failed

Root causes:
- `/health` reports `mock_mode` from `settings.is_mock_mode()` (key present ⇒
  "not mock" ⇒ frontend shows `LIVE — GROQ`). No code path ever verifies the
  provider works; the badge lied about health, not about configuration.
- `reports/judge_human_agreement.json` records the truth: 50/50 live judge calls
  failed (`n_failed: 50`). Root cause found in code: both `scripts/run_live_judge.py`
  and `scripts/compare_judge_to_human_fast.py` call `response.parse_error` on the
  provider's `LLMResponse`, which has no such attribute → every successful API call
  raised `AttributeError`, was swallowed by a bare `except`, and counted as failed.
  **The provider may have been working all along — the harness had the bug.**

Fix: distinguish CONFIGURED vs HEALTHY/VERIFIED end-to-end (new provider-status
module + `/api/v1/provider/health`), fix the judge scripts, run the real test
(`scripts/test_llm_provider.py`), and render UNHEALTHY/CONFIGURED states in the UI.

### BUG 4 (MEDIUM) — Failure-analysis provenance drift

The Failure Analysis page described "93 failures" (the old report); the current
report (regenerated against the remapped dataset) contains 161. The page text also
hardcoded an assumption ("human-verified golden set") that is true of the labels but
not of the classifier being evaluated post-remap. Fixed by sourcing provenance from
the report data itself and adding a data-as-of note (see BUG 1 for the underlying fix).

### BUG 5 (LOW) — judge endpoints' VALIDATED logic

`/api/v1/llm-judge/summary` can flip to VALIDATED for live-judge runs whose "human"
scores are proxy scores derived from golden labels (`scripts/score_human_proxy.py`),
which are not human ratings. VALIDATED now additionally requires
`human_scoring_method` to be a real human study; proxy-scored agreement renders as
"PIPELINE VALIDATED — HUMAN AGREEMENT NOT AVAILABLE".

### BUG 6 (LOW) — Startup/env hygiene

- `docker-compose.yml` sets `APP_ENV=production` with `CORS_ORIGINS=*` in the local
  `.env` (refused by the backend in production — would break the compose stack).
  `.env.example` now ships explicit origins and documents the interaction.
- `MODEL_NAME` is documented; `GROQ_MODEL` added as an explicit alias in the
  factory; `LLM_MODE` documented as the explicit mock/live intent flag.

## 3. Provider status model (implemented)

`backend/app/providers/llm/status.py` distinguishes:

| State | Meaning |
|---|---|
| `mock` | provider=mock in use; no external calls |
| `not_configured` | live provider selected but no API key |
| `configured` | key present; not yet verified |
| `healthy` | real minimal API call succeeded (model listed; tiny completion OK) |
| `unhealthy` | key present but the verification call failed (sanitized error code) |

Surfaced via `/api/v1/provider/health` (and folded into `/api/v1/system`). The
frontend never derives provider state from env names; it renders what the backend
measured. Secrets never leave the backend.

## 4. Verification status

See `FINAL_VERIFICATION.md` (updated with the post-fix runs). The golden-set
regeneration outcome and the real Groq API test results are recorded there and in
`reports/golden_gap_analysis.md`.
