# Architecture Audit — 2026-09-16

Scope: full-repository inspection (backend, frontend, scripts, reports, data,
Docker, tests, docs) before the redesign pass. Every behavioral claim below was
**executed and observed**, not assumed: the 10 assignment probe messages were
run through the live API, and the current artifact numbers were read from
`reports/*.json`. Nothing here is projected.

---

## 1. What the system currently does

An **evidence-first support decision system** for AmazonHelp (brand selected
from data, `reports/brand_profile.md`). A customer message flows through:

```
Customer message
  → Intent classification (TF-IDF + LogisticRegression, 11-intent taxonomy in the artifacts)
  → Historical case retrieval (TF-IDF cosine, temporal index, k≤20)
  → Evidence quality scoring (empirically calibrated weights)
  → Response generation (LLM, evidence-only prompt, must abstain if insufficient)
  → Claim verification (word-overlap against the evidence text, independent of the LLM)
  → Risk/escalation policy (multi-signal, named reason codes)
  → AUTO / ESCALATE + inspectable audit trail
```

**The core product principle is implemented and enforced in code:** when
evidence is insufficient the system abstains and escalates rather than
inventing an answer. It is a decision system with an audit trail, not a
chatbot — no free chat surface exists anywhere in the product.

## 2. Request lifecycle (verified live)

`POST /api/v1/agent/respond` → `SupportAgent.respond()`
(`backend/app/services/agent.py`):

1. classify → intent + full probability distribution (2–4 ms)
2. retrieve k cases (TF-IDF cosine, temporal index)
3. score evidence (calibrated weights over retrieval-only features)
4. compute ambiguity/novelty signals (`services/ambiguity.py`)
5. generate draft (provider via `providers/llm/factory.py`); provider failure
   → `GENERATION_FAILED` signal, never a fabricated draft
6. ground/verify claims (`generation/generator.py::check_grounding`)
7. policy decides (`escalation/policy.py`), fail-safes first

Cross-cutting: request IDs (`X-Request-ID`, echoed in bodies and structured
logs), one error shape `{error: {code, message, request_id}}`, PII-free
structured decision logs, per-stage latency, rate limiting on agent endpoints.

## 3. Frontend architecture

Static HTML/CSS/JS, no build step, served by Flask (and nginx in Docker).
App shell with hash routing (`frontend/js/app.js`), 8 pages, every number
fetched from the API (grep-verified: no hardcoded report values). UI toolkit
(`frontend/js/ui.js`): `el()`, badges, stat blocks, skeletons, error/empty
boxes, CSS-only tooltips.

Honesty mechanisms already in place: persistent mode pill that distinguishes
MOCK / LIVE(healthy) / UNHEALTHY / UNVERIFIED / NOT CONFIGURED; mock-draft
warning banner; "NOT VALIDATED" judge stamp; silver vs golden separation
everywhere; "what the headline number hides" section.

## 4. Backend architecture

- Flask + gunicorn (FastAPI/Pydantic deliberately not installed — decision
  #13; the port is mechanical and documented per-endpoint in docstrings).
- Provider abstraction: `LLMProvider` base → `GroqProvider` (live-verified
  2026-09-16), `GeminiProvider` (structurally complete, unexercised),
  `MockLLMProvider` (deterministic, tagged `is_mock=True`).
- Provider status (`providers/llm/status.py`): distinguishes configured vs
  **measured** health (`/api/v1/provider/health?verify=1` performs a real
  models-list + 1-token completion). The UI renders measured state only.
- Silent mock fallback exists by design for missing keys but is *visible*:
  `/health` reports `mock_mode`, the factory logs a warning, the UI shows the
  MOCK pill. (In `APP_ENV=production` an operator can force failure instead:
  see §10.)

## 5. ML pipeline

- Taxonomy: 12 configured intents; **the artifacts were trained after the
  2026-09-12 remap** — `account_access_issue` has ~6 training examples and is
  effectively absent from the silver test set (11 intents with support).
- Classifier: TF-IDF (1–2 grams, 20k features) + multinomial LR
  (class_weight=balanced). Returns full distribution → margin signals.
- Silver labels are cluster-derived (circularity documented); golden set is
  200 human-verified examples (single labeler, documented).

## 6. Retrieval pipeline

TF-IDF cosine over ~17k resolved conversations, index built **only** from
splits strictly earlier than the query period (enforced structurally in
`retrieval/retriever.py`). Recall@1 = 0.495, Recall@5 = 0.833, MRR = 0.617
(`reports/retrieval_metrics.json`).

## 7. Generation pipeline

Structured JSON prompt (system policy / customer message / evidence as
separate labeled fields — prompt-injection defense), strict JSON output schema
(`draft_reply`, `grounded_claims`, `unsupported_claims`, `confidence`,
`evidence_insufficient`), mandatory abstention path, `max_tokens=2000`
(reasoning models spend tokens before the JSON), retries with backoff on 429,
immediate failure on 4xx.

## 8. Grounding pipeline

`check_grounding()` verifies the LLM's claimed grounded statements against
evidence text with a ≥60% word-overlap rule. Deliberately literal
(semantic entailment needs a real LLM — documented limitation). Mock provider
abstention path returns grounded=True with an explanation.

## 9. Evaluation methodology (current, honest)

| Number | Value | Source |
|---|---|---|
| Silver accuracy (cluster-derived labels) | 93.57% | `baseline_results.json` |
| Golden accuracy (human-verified) | **19.5%** | `golden_set_evaluation.json` |
| Gap | 74.07 points | same |
| Golden macro-F1 | 0.182 | same |
| Silver auto-precision / coverage | 99.0% / 20.1% | `automation_curve.json` |
| Golden auto-precision / coverage | 68.75% (CI 51.4–82.1) / 16.0% | `golden_automation_check.json` |
| Recall@1 / @5 / MRR | 49.5% / 83.3% / 0.617 | `retrieval_metrics.json` |
| Leakage: random vs temporal split | 2,701 vs 16 dup groups | `leakage_analysis.json` |
| Judge | NOT VALIDATED (live run vs proxy-human scores) | `judge_human_agreement.json` |

Both silver and golden are reported side-by-side everywhere; the UI never
quotes one alone.

## 10. Provider behavior — verified truthful

Tested with `?verify=1` on 2026-09-16: `{"provider":"groq","mode":"live",
"configured":true,"healthy":true,"model":"openai/gpt-oss-120b",
"latency_ms":1173}`. A full live agent request passed (`generation_status:
PASS`, provenance `groq/live`). The UI can only show LIVE after a measured
check; a failed check shows UNHEALTHY with the sanitized error code.

Remaining honest gap: in `APP_ENV=production`, a missing key currently falls
back to mock visibly rather than failing startup. Acceptable for a demo
(visible everywhere), but fail-fast would be stricter — recorded as debt.

## 11–14. Data / API / mock-live behavior

Data pipeline: 2.81M-row Kaggle corpus → streamed graph index → seeded 60k
AmazonHelp sample → clustering → silver labels → temporal split → baselines →
retrieval metrics → evidence calibration → automation sweep → golden set →
failure analysis → saved artifacts. Every report has a committed generating
script; nothing hand-typed.

API: 15 endpoints, all smoke-tested; validation, error shape, request IDs,
rate limit, CORS (wildcard only in dev), gunicorn in production.

Mock/live: mock is deterministic, abstains with no evidence, tagged
`is_mock=True`, and its judge scores are deliberately uniform so they can
never pass as a real evaluation. Live Groq verified as above.

## 15. Current bugs (verified by running the 10 assignment probe cases)

Measured 2026-09-16 via the live API:

| # | Probe | Observed | Verdict |
|---|---|---|---|
| 1 | "My package says delivered but I never received it." | `delivery_not_received` @ 1.00 | Correct intent; confidence is the model's genuine softmax output (documented TF-IDF saturation, not a UI bug) |
| 2 | "Where is my package? It was supposed to arrive yesterday." | `delivery_not_received` @ 0.986 | Wrong-ish (delay fits); margin signal present |
| 3 | "I requested a refund three days ago..." | `cancellation_or_refund_request` @ 0.974 | Correct |
| 4 | "I was charged twice for my order." | `general_other` @ 0.559 | **BUG** — payment intent exists but has 19 training examples; real retrieval disagreement missed |
| 5 | "The app crashes every time..." | `app_or_device_technical_issue` @ 0.996 | Correct |
| 6 | "My account was hacked and I cannot log in." | `general_other` @ 0.802 | **BUG** (data: intent collapsed post-remap) — escalates safely but for the wrong reason |
| 7 | "What is the capital of India?" | `content_availability_inquiry` @ **1.00** | **BUG** — confident OOD misclassification; only caught downstream by weak evidence |
| 8 | "I need help." | `general_other` @ 0.851 | OK (abstention bucket) |
| 9 | Multi-issue message | `MULTI_INTENT_SUSPECTED` → ESCALATE | **Works** (multi-intent veto validated) |
| 10 | Prompt-injection probe | safe handling, no leakage; test coverage exists | Works |

### Root causes and fixes applied in this pass

- **BUG A — OOD overconfidence (probe 7):** no novelty signal existed at
  classification time. Fix: `services/novelty.py` computes an OOD score from
  measured quantities (top-1 probability, top-2 margin, top retrieval
  similarity, retrieval-intent agreement) — calibrated on the golden set, not
  keyword matching — and the policy gains an `OOD_REQUEST` reason code.
- **BUG B — Retrieval disagreement surfaced only as a weak-corroboration
  flag:** classifier-vs-retrieval-majority disagreement is now its own named
  `RETRIEVAL_DISAGREEMENT` reason code (probe 4's evidence majority was
  `general_other` while the classifier drifted elsewhere — the disagreement
  is real and should be explicit).
- **BUG C — Risk reasons too coarse:** `general_other` was implicitly
  high-risk via its `escalation_tendency`, but the *reason* shown was
  evidence-threshold noise. The policy now emits explicit
  `LOW_CLASSIFICATION_CONFIDENCE`, `WEAK_EVIDENCE`, `OOD_REQUEST`,
  `RETRIEVAL_DISAGREEMENT`, `MULTI_INTENT`, `UNSUPPORTED_CLAIMS`,
  `GENERATION_FAILED`, `PRIVACY_RISK` codes; `HIGH_RISK_INTENT` remains only
  for the four financial/security intents.
- **BUG D — Claim/evidence mismatch risk:** grounding verified only the
  LLM's *self-reported* `grounded_claims` list; claims shown in the UI could
  in principle not correspond to the draft text, and unsupported claims were
  reported but not per-claim verified against evidence. Fix: claims are now
  extracted from the **actual draft text** (split on sentence boundaries),
  each claim independently verified against evidence with matched evidence
  IDs, and the API returns per-claim objects
  `{claim_text, status, evidence_ids, confidence, explanation}`. The old
  self-reported lists remain available for comparison but never drive the UI.

## 16. Misleading UI claims — status

None found in the current frontend: all metrics fetched, silver/golden
labeled, judge NOT VALIDATED, mock visibly tagged, provider state measured.
The remaining risk was the claim-verification display (BUG D) — fixed with
draft-derived claims.

## 17. Technical debt (top items; full list in TECHNICAL_DEBT.md)

1. TF-IDF semantic ceiling (Recall@1 49.5%) — embeddings are the known fix.
2. Classifier degraded by the remap (golden 19.5%) — retrain from pre-remap
   mapping is the real fix but changes all committed reports; recorded as the
   documented finding instead of silently "improving" numbers.
3. Golden automation precision CI ±15 points (n=32).
4. Grounding is word-overlap, not entailment.
5. Evidence-score calibration targets classifier correctness, not groundedness.

## 18. Production risks

Docker smoke-verified only; in-memory rate limit (per-process); no persistent
decision store (request IDs correlate to logs, not a DB); model artifacts
pin the exact sklearn version; single labeler.

## 19. What should be preserved

- The multi-signal escalation policy with named reason codes.
- Silver/golden side-by-side honesty; the "headline number hides" framing.
- Provider status truthfulness (measured health, sanitized errors).
- The evidence scorer's calibrated weights; temporal retrieval discipline.
- Tests (112 passing) and the failure-analysis-from-real-data approach.

## 20. What should be redesigned (this pass)

1. **Live Agent UX** — currently a three-pane developer console; becomes a
   40/60 agent workspace with a horizontal decision timeline, a hero decision
   card, per-claim verification, and a human-readable "why" narrative.
2. **Claim verification** — draft-derived claims with per-claim evidence IDs
   (BUG D).
3. **OOD/novelty signal** — `OOD_REQUEST` reason code (BUG A).
4. **Explicit retrieval disagreement** (BUG B) and richer reason codes (BUG C).
5. **Docs** — ARCHITECTURE_AUDIT (this file), known-limitations, api reference,
   reproducibility already exists; interview guide refresh.
6. Progressive pipeline states bound to real stage latencies; accessibility
   and responsive polish.

Out of scope (deliberately): FastAPI/Pydantic port, embedding retrieval,
retraining from the pre-remap mapping, and any change that would move
committed report numbers without re-running their generating scripts.
