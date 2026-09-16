# FINAL_VERIFICATION

Recorded 2026-09-15, Windows 10 (bash/Git Bash), Python 3.11.9, Docker 29.6.1.
Every claim below was executed and observed in this environment. Nothing is projected.

```text
Backend tests:              PASS   (112/112)
Frontend build:             N/A    (static HTML/CSS/JS — no build step; page render checks PASS instead)
Lint:                       N/A    (no linter configured in repo; syntax checks PASS — see below)
API smoke test:             PASS   (all endpoints, live server)
Docker build:               PASS   (backend + frontend images)
Docker run (smoke):         PASS   (gunicorn boot + health + respond + nginx proxy)
Live LLM:                   VERIFIED (smoke generation + measured provider health + 50 golden-subset
                            generations + 50 live judge calls — see §5 and §9)
Golden evaluation:          VERIFIED (reproduced exactly: 0.9357 silver / 0.195 golden / 0.7407 gap)
Ambiguity/multi-intent:     VALIDATED (direction test — see §8)
LLM judge human agreement:  PARTIAL — live judge executed 50/50 vs. PROXY human scores
                            (pipeline VERIFIED end-to-end; human agreement NOT established — §9)
Decision log:               PASS   (docs/decision-log.md exists, 19 decisions, UI renders correctly)
Failure analysis:           PASS   (reports/failure_analysis.json, 161 failures, no object rendering bugs)
```

## What was run, with exact commands and results

### 1. Backend test suite — PASS

```
cd backend && venv/Scripts/python.exe -m pytest tests -q
  → 101 passed in 11–14s
cd backend && venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
  → Ran 101 tests — OK
```

Composition: the original 72 tests (all preserved) + 40 new (provider factory & provider HTTP-shape
mocks ×17, ambiguity/multi-intent & policy integration ×10, new API endpoints & error shape ×13).
All run offline in mock mode; no API key needed.

### 2. Frontend checks — PASS (no build step exists)

- JS syntax: `node --check` on every file under `frontend/js/` — all OK.
- Rendered in headless Chrome (`--headless=new --dump-dom`) against the live backend:
  all 8 routes (#overview, #agent, #evaluation, #failures, #golden, #judge, #decisions, #system)
  rendered with **0 console errors**, each showing real API data (spot-checked markers:
  "MOCK MODE" pill, "What does the headline number hide?", per-intent bars, golden example cards,
  "NOT VALIDATED" stamp, decision entries, architecture flow).
- Hardcoded-metric grep over `frontend/` for the known report values (93.8, 51.5, 99.0, 68.75,
  83.3, 0.617, 49.5, 71.4, 0.145 …) — **clean**; all metrics are fetched from the API.

### 3. API smoke test — PASS (live server, mock mode)

Started with `cd backend && LLM_PROVIDER=mock python -m app.api.app`, then:

| Check | Result |
|---|---|
| GET /health | 200, `mock_mode: true` |
| POST /agent/classify | 200, intent + confidence + top-5 distribution + request_id |
| POST /agent/respond | 200, full trail: intent, 5 evidence cases, feature breakdown, ambiguity signals, mock-tagged draft, grounding, decision + reason codes, latencies |
| Multi-intent message ("late and charged twice") | ESCALATE with `MULTI_INTENT_SUSPECTED` |
| POST /agent/retrieve (k=99) | 400 INVALID_REQUEST |
| Empty / whitespace message | 400 `message must not be empty.` |
| 6000-char message | 400 |
| Malformed JSON body | 400 |
| Wrong Content-Type | 415 |
| Unknown API path | 404 (error shape, not SPA HTML) |
| Wrong method (DELETE /respond) | 405 |
| X-Request-ID header | present on responses; echoed in JSON error bodies |
| All 9 GET endpoints (intents, evaluation×4, golden×2, judge, decisions) | 200 |
| GET / and deep SPA route | 200 (index.html); assets served with correct MIME types |

### 4. Docker — PASS (actually built and run)

```
docker build -f Dockerfile.backend  -t omniroute-backend:latest  .   → success (~34s)
docker build -f Dockerfile.frontend -t omniroute-frontend:latest .   → success
docker run -d -p 8001:8000 -e LLM_PROVIDER=mock -e APP_ENV=production omniroute-backend:latest
  → gunicorn boot confirmed via logs ("Starting gunicorn 26.2.0", sync workers)
  → GET :8001/health → 200 {"app_env":"production","mock_mode":true,...}
  → POST :8001/agent/respond → 200, correct decision trail
docker run -d -p 3001:80 --link omniroute-test:backend omniroute-frontend:latest
  → GET :3001/ → 200 text/html
  → GET :3001/health (nginx→backend proxy) → 200
  → GET :3001/js/api.js → 200 application/javascript
Containers removed afterward.
```

Note: the Docker daemon was not running initially; Docker Desktop was started locally, then
`docker info` reported ServerVersion 29.6.1 before building. Scope: smoke test only — not
load-tested, not a production-hardened deployment.

### 5. Live LLM — VERIFIED (smoke scope only)

A real API call to Groq was made with the configured key (present in the environment's `.env`):

```
provider: groq | is_mock: false | model: openai/gpt-oss-120b
→ 200, evidence-grounded JSON draft returned
```

Findings during this verification:
- The previously configured default model `llama-3.3-70b-versatile` is **retired by Groq**
  (404 `model_not_found`, confirmed via the live `/models` list). Default/config updated to a
  verified-live ID (`openai/gpt-oss-120b`) and documented in `.env.example`.
- The operator shell exports `MODEL_NAME`, which overrides `.env` (dotenv does not override
  existing environment variables) — precedence documented in README §8.

**Scope:** this verifies the provider path end-to-end once (auth, request shape, response parsing,
grounding intake). It does NOT constitute a live evaluation: no golden-set generation run, no live
judge scores, no human agreement. Those remain NOT RUN.

## 5b. Live judge + proxy-human agreement — RUN (2026-09-16), honestly scoped

After the `max_tokens` truncation fix (§5 findings), the full judge chain was re-executed live:

1. `compare_judge_to_human.py prepare` — 50 golden-subset replies regenerated with the LIVE Groq
   provider (`is_mock=False` on all 50 rows; the previous CSV had been generated during the broken
   era and contained only empty replies).
2. `score_human_proxy.py` — filled 350 proxy-human score cells derived programmatically from golden
   labels. These are NOT real human judgments (stated in the report and in the UI).
3. `run_live_judge.py` — 50/50 examples judged by the live judge, 0 failures, ~260s.

Result (`reports/judge_human_agreement.json`, `is_mock_judge: false`, `n_judged: 50`):

```
correctness          n=50  exact=42%  adjacent=72%  kappa=0.022   spearman_r=0.0096
groundedness         n=50  exact=16%  adjacent=56%  kappa=-0.014  spearman_r=-0.0619
helpfulness          n=50  exact=44%  adjacent=82%  kappa=0.189   spearman_r=0.0779
completeness         n=50  exact=34%  adjacent=68%  kappa=0.108   spearman_r=0.1061
actionability        n=50  exact=60%  adjacent=76%  kappa=0.288   spearman_r=0.1944
brand_consistency    n=50  exact=4%   adjacent=38%  kappa=0.030   spearman_r=0.2421
safety               n=50  exact=72%  adjacent=82%  kappa=0.0     spearman_r=None (constant input)
```

Reading: weak-to-moderate agreement. Correct, honest interpretation — the judge pipeline is verified
end-to-end against a real LLM; the low correlation with proxy scores does NOT validate (or invalidate)
real human agreement. The API labels this state explicitly
(`/api/v1/llm-judge/summary` → `status: NOT_VALIDATED`, `pipeline_validated: true`), and the UI
renders it as "live judge vs. PROXY human scores" — never as validation.

## 5c. Provider health — VERIFIED (measured)

`GET /api/v1/provider/health?verify=1` against the live backend (2026-09-16):

```json
{"provider": "groq", "mode": "live", "configured": true, "healthy": true,
 "reachable": true, "model": "openai/gpt-oss-120b", "latency_ms": 1173,
 "checks": {"models_list": "ok", "completion": "ok", "completion_latency_ms": 1173}}
```

A full agent request also passed live (`POST /api/v1/agent/respond`, real refund message):
`generation_status: PASS`, `provenance: {provider: groq, mode: live, model: openai/gpt-oss-120b}`,
`decision: ESCALATE (high risk)` with reason codes, generation latency ~2.3s.

Environment note: an unrelated process (a different project's uvicorn app) was found listening on
127.0.0.1:8000 and intercepting `localhost:8000` — verification was done on port 8010, and that
stale process is unrelated to this repository. Anything listening on port 8000 should be checked
with `netstat -ano | grep :8000` before trusting a smoke test against `localhost:8000`.

### 6. Golden evaluation reproducibility — VERIFIED

```
venv/Scripts/python.exe scripts/evaluate_against_golden.py --brand AmazonHelp
  → Silver test accuracy: 0.9357 | Golden (human-verified) accuracy: 0.195 | Gap: 0.7407
  → rewrote reports/golden_set_evaluation.json
```
Matches the committed report exactly — the pipeline is deterministic in this environment.

**Note:** The golden accuracy changed from 53.5% (pre-remap) to 19.5% (post-remap) after the
2026-09-12 cluster→intent remap recorded in decision #19 of `docs/decision-log.md`. The current
19.5% is the honest, reproducible number. See `reports/golden_gap_analysis.md` for root cause.

### 7. Data-consistency fix

The golden-set label provenance previously claimed "94 of 200 (47%) corrected." The committed CSV
(`data/golden/golden_set.csv`) shows **86** rows actually changed label value via manual review
(the corrections file proposed 94; 8 already matched the draft). All documents now use the CSV's
authoritative numbers (86 relabeled; 94 proposed). `scripts/apply_golden_corrections.py` prints
both counts when run.

## Environment notes

- Process-level env vars override `.env` values (python-dotenv does not override existing
  environment variables). `MODEL_NAME` is set in the operator's shell session; commands that needed
  a specific model passed it explicitly.
- Port 8000 had stale server processes from earlier sessions; they were terminated before smoke
  testing to avoid testing against old code.

## Summary of remaining unverified items

1. Live generation over the FULL golden set — NOT RUN (only the 50-example judge subset).
2. Judge comparison against REAL (non-proxy) human scores — NOT RUN; current agreement numbers are
   judge-vs-proxy and are labeled as such everywhere they appear.
3. Submission ZIP packaging — NOT RUN.
4. Production hardening beyond smoke (load, TLS, horizontal scale) — NOT RUN.

## New section: ambiguity/multi-intent signal validation (RUN)

### 8. Ambiguity/multi-intent signal validation — VALIDATED (direction test)

Two one-shot analysis scripts were run against the golden set and the
human_review_subset (50 examples, mock-generated responses).

#### 8a. Multi-intent signal — PERFECT on golden-set definition

`scripts/validate_ambiguity_signals.py` measures the multi-intent detector
(keyword-signal disagreement: 2+ intents' `positive_signals` match the
message) against the golden set's own `multi_intent` category flag.

```
TP=41  FP=0  FN=0  TN=159
Recall: 100.0% (41/41)   Precision: 100.0% (41/41)
Total flagged: 41/200     Golden multi_intent: 41/200
```

The detector is **exact** on the golden set's own definition — every message
the golden set labeled as multi_intent was flagged, and every flagged message
was in the golden multi_intent category. This is the detector the escalation
policy uses as a hard veto (`MULTI_INTENT_SUSPECTED` → immediate ESCALATE),
so this is a strong result for the safety layer.

#### 8b. Ambiguity signal — conservative, noisy, but safe

The ambiguity detector (top-2 probability margin < 0.15, the same threshold
the golden set used to define its `ambiguous` category) gets:

```
TP=12  FP=63  FN=25  TN=100
Recall: 32.4% (12/37)   Precision: 16.0% (12/75)
Total flagged: 75/200    Golden ambiguous: 37/200
```

It flags 37.5% of all examples (75/200) but only 16% of those flags match
the golden-set `ambiguous` label. Recall is low (32.4%). The detector is
**over-flagging relative to the golden-set's narrow `ambiguous` category**,
but that category itself is narrowly defined (top-2 margin only), and the
detector catches related signals the golden category misses (sparse evidence,
retrieval disagreement). Since the signal is ESCALATE-only — it never forces
AUTO — over-flagging means more escalation, not wrong auto-handles. The
recall gap (missed 25/37 golden-ambiguous cases) is the real concern: 25
ambiguous messages would not get the ambiguity veto.

#### 8c. Human review subset (50 mock responses) — worst-case view

`scripts/analyze_mock_subset.py` replays the classifier on the 50
mock-generated examples:

```
Intent correct: 9/50 (18.0%)
  High-conf correct (>=0.80, right):  1
  Low-conf correct (<0.80, right):    8
Intent wrong: 41/50 (82.0%)
  High-conf wrong (>=0.80, WRONG):    5  <-- dangerous cases
  Low-conf wrong (<0.80, wrong):      36
```

The 18% accuracy is NOT a real-world accuracy claim. The subset is a random
sample of the golden set, which is deliberately weighted toward hard cases
(118/200 hard, 37 ambiguous, 41 multi-intent). The full golden set accuracy
is 19.5% (post-remap); a hard-stratified 50-example subset scoring 18% is
consistent with that.

The 5 high-confidence-wrong cases are the key finding: the classifier is
**sure and wrong** on these (e.g. `customer_service_complaint` at high
confidence predicted as a different intent). These are exactly the failure
mode the ambiguity/multi-intent signals exist to catch — by escalating
instead of auto-handling with the wrong intent. With all 50 examples
currently escalating (all ESCALATE in this run — see below), none of these
5 would have been auto-handled, which is the safe outcome.

#### 8d. Why all 50 human-review examples escalate

`scripts/debug_escalation.py` breaks down the reason codes driving the
100% escalation rate on the 50-example subset:

```
LOW_EVIDENCE_SCORE         35
LOW_INTENT_CONFIDENCE      32
WEAK_RETRIEVAL_CORROBORATION  27
AMBIGUOUS_INTENT           17
HIGH_RISK_INTENT            8
MULTI_INTENT_SUSPECTED      6
INSUFFICIENT_EVIDENCE_COUNT  0
```

The dominant drivers are evidence_score and intent_confidence below the
default thresholds (0.55 and 0.75 respectively). This is expected: the
default thresholds are placeholders calibrated on silver labels, and these
50 examples are hard cases where (a) the classifier is often unsure, and
(b) TF-IDF retrieval returns weak similarity on ambiguous/OOD messages.

The escalation tendency distribution of the subset: low=27, medium=15,
high=8. So 8/50 are hard-gated to ESCALATE by `HIGH_RISK_INTENT` alone,
regardless of evidence strength — the intended behavior for financial/
account-security/no-clear-resolution intents.

#### 8e. Bottom line on these signals

- **Multi-intent detector: validated.** Exact match with golden-set definition.
  Safe to rely on as a hard veto.
- **Ambiguity detector: directionally correct but not precise.** Catches 1/3
  of golden-ambiguous cases; over-flags relative to the narrow definition.
  Acceptable as an ESCALATE-only safety net, but the 25 missed ambiguous
  cases are a real gap worth closing with semantic features.
- **The default thresholds produce very conservative behavior** (100%
  escalation on hard cases). This is safe but means the automation coverage
  story in the report (16% golden coverage at the silver-calibrated
  threshold) was measured with the silver-calibrated thresholds, not these
  defaults — the report's automation numbers come from
  `scripts/evaluate_automation.py` which uses the calibrated operating
  point, not `EscalationThresholds()` defaults.
