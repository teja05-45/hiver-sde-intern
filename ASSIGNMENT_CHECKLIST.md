# Assignment Checklist

Legend: ✅ implemented & verified in this environment | ⚠️ implemented, verification incomplete/partial | ❌ not done

| Requirement | Status | Evidence | File(s) | Verification |
|---|---|---|---|---|
| Pick one brand from dataset | ✅ | Data-driven selection, scoring bug caught & fixed | `scripts/profile_dataset.py`, `reports/brand_profile.md` | Pipeline run in this repo |
| Intent classification | ✅ | 12 intents, real taxonomy | `configs/intents.yaml`, `backend/app/classification/` | 101 tests pass |
| Grounded reply drafting | ✅ structure / ⚠️ quality | Evidence-only prompt, mandatory abstention | `backend/app/generation/generator.py` | Mock end-to-end + LIVE Groq smoke call verified; quality eval NOT RUN |
| AUTO/ESCALATE decision + reason | ✅ | Multi-signal policy, named reason codes | `backend/app/escalation/policy.py` | Unit-tested; live API verified |
| Runnable pipeline | ✅ | Seeded, one script per stage | `scripts/*.py` | Golden evaluation re-run reproduced 0.9378/0.535 exactly |
| Golden set 150–250 examples | ✅ | 200 examples, stratified 8 categories | `data/golden/golden_set.csv` | Committed artifact |
| Golden set methodology explained | ✅ | Sampling, labeling, 86 relabeled + 94 proposed corrections, single-labeler limitation | `data/golden/README.md` | — |
| Evaluation harness | ✅ | Accuracy/P/R/F1/confusion from scratch | `backend/app/evaluation/metrics.py` | Unit-tested |
| LLM-as-judge rubric | ✅ structure / ⚠️ NOT VALIDATED | 7 dimensions, structured output | `backend/app/evaluation/judge.py` | Mock only; UI shows NOT VALIDATED stamp |
| Judge–human agreement | ⚠️ | Spearman/weighted-κ/exact-adjacent implemented | `backend/app/evaluation/agreement.py` | Unit-tested; no human data collected (honestly labeled) |
| Final report | ✅ | Required sections incl. literal "What is misleading about my headline number?" | `reports/final_report.md` (REPORT.md is a stale earlier draft) | — |
| Reproducible subsample | ✅ | 60k seeded sample of 2.81M | `scripts/build_dataset.py` | — |
| No fabricated metrics | ✅ | Every number traces to a script; UI fetches all metrics from API (grep-verified, no hardcoded values) | `reports/*.json`, `frontend/js/` | grep check clean |
| Auto-precision ≥99% honestly reported | ✅ | 99% silver / 68.75% golden (95% CI 51.4–82.1%) side by side everywhere | `reports/automation_curve.json`, `reports/golden_automation_check.json` | — |
| Selective-coverage precision vs accuracy distinguished | ✅ | Throughout README/report/UI | Multiple | — |
| REST API | ✅ | 15 endpoints incl. evaluation/golden/judge/decisions/system | `backend/app/api/app.py` | Live smoke test: all endpoints 200/expected codes |
| API validation & error shape | ✅ | One error shape, request IDs, empty/long/invalid-JSON/wrong-type/k-range handled | `backend/app/api/app.py` | Live smoke test (400/404/405/415 verified) |
| Request ID + structured logging | ✅ | X-Request-ID header, JSON log lines w/ decision metrics, no PII/secrets | `backend/app/api/app.py` | Verified live |
| CORS configurable, no wildcard in prod | ✅ | Env-driven; wildcard only honored in APP_ENV=development | `backend/app/api/app.py`, `.env.example` | — |
| Production WSGI server | ✅ | gunicorn w/ worker/timeout/graceful config | `backend/wsgi.py`, `backend/gunicorn.conf.py`, `Dockerfile.backend` | Container ran under gunicorn |
| LLM provider configurable via env, no hardcoded keys | ✅ | `LLM_PROVIDER`, factory, `.env.example` | `backend/app/providers/llm/factory.py` | Unit-tested (mock/groq/gemini/invalid/missing-key) |
| Deterministic/mock mode | ✅ | Mock provider tagged `is_mock`; persistent MOCK MODE indicator in UI | `backend/app/providers/llm/mock_provider.py` | 101 tests run offline |
| Live LLM | ⚠️ smoke-verified | Real Groq API call succeeded (real key, `openai/gpt-oss-120b`); retired default model ID fixed | `FINAL_VERIFICATION.md` | Generation smoke ✅; full live evaluation NOT RUN |
| Python/FastAPI/Pydantic preferred stack | ⚠️ | Flask chosen (see decision-log #13); port is mechanical | `docs/decision-log.md` | Fully tested as-is |
| Test suite without API key | ✅ | 101 tests (72 original preserved + 29 new) | `backend/tests/` | pytest + unittest: 101 passed |
| White-theme professional UI | ✅ | Operations console: white theme, 8 pages, no build step | `frontend/` | Rendered in headless Chrome, 0 console errors, all pages |
| UI pages (overview, live agent, evaluation, failures, golden set, judge, decisions, system) | ✅ | Persistent shell, sidebar, mock/live pill, "What does the headline number hide?" section | `frontend/js/pages/` | Browser-verified per page |
| Data pipeline (streaming, dedup, validation) | ✅ | Streaming CSV, dedup analysis, validation stats | `backend/app/services/` | — |
| Leakage prevention | ✅ | Temporal split + measured duplicate overlap | `reports/leakage_analysis.json` | — |
| ≥2 baselines | ✅ | Majority + TF-IDF/LogReg | `backend/app/classification/baselines.py` | — |
| Retrieval + Recall@K/MRR | ✅ | Temporal index rule enforced structurally | `backend/app/retrieval/` | — |
| Escalation engine, multi-signal | ✅ | + experimental ambiguity/multi-intent vetoes (ESCALATE-only) | `backend/app/escalation/policy.py`, `backend/app/services/ambiguity.py` | Unit-tested; live-verified MULTI_INTENT_SUSPECTED |
| Docker (backend, frontend, compose) | ✅ | Images built; backend ran under gunicorn; nginx proxy verified | `Dockerfile.*`, `docker-compose.yml` | REAL BUILD + RUN this environment — see FINAL_VERIFICATION.md |
| Decision log (10–15 decisions) | ✅ | 16 decisions incl. new ambiguity & resolution-consistency entries | `docs/decision-log.md` | Rendered via /api/v1/decisions |
| Interview prep | ✅ | 30s/2min/5min + interviewer Q&A | `docs/interview-guide.md` | — |
| Technical debt doc | ✅ | Real issues only | `TECHNICAL_DEBT.md` | — |
| Submission ZIP | ❌ | Not packaged in this environment — see FINAL_VERIFICATION.md | — | NOT RUN |

## Dataset & tooling citation

- Dataset: "Customer Support on Twitter," Kaggle, `thoughtvector/customer-support-on-twitter`.
- Libraries: scikit-learn, pandas, numpy, scipy, Flask, gunicorn, PyYAML, joblib (standard
  open-source, per their licenses; no code copied from their source).
- No external code, text, or methodology copied without attribution; prompts, scoring formulas,
  and architecture were written for this project.

## Honest summary of what is still NOT verified

1. **Full live-LLM evaluation** — generation over the golden set, live judge scores, and
   judge–human agreement were NOT run (one real Groq API call was verified; that's all).
2. **Ambiguity/multi-intent signals** — direction-tested in unit tests, not validated against a
   labeled ground truth (labeled EXPERIMENTAL everywhere they appear).
3. **Production hardening** — Docker smoke ≠ load-tested, secured, or horizontally-scaled deployment.
4. **Submission packaging** — no ZIP was produced in this environment.
