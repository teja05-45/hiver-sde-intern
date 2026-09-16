# Audit Before UI Redesign — EvidenceDesk

**Date:** 2026-09-16 · **Branch:** `main` · **Scope:** pre-redesign audit of the existing
"Evidence-First Customer Support Agent" (Hiver SDE Intern take-home).

The UI is being redesigned from an ML/evaluation dashboard into a support-operations product.
This document records what exists today, what is preserved, and what changes. **No working
backend/ML functionality is removed.**

---

## 1. Existing architecture

```
Customer message
  → Intent classification (TF-IDF + LogisticRegression, 12-intent taxonomy)
  → Historical evidence retrieval (TF-IDF cosine + hybrid intent-aware reranking,
    temporal index over ~17k resolved AmazonHelp conversations)
  → Evidence quality scoring (empirically calibrated feature weights)
  → Grounded draft generation (LLM via provider abstraction: groq | gemini | mock,
    evidence-only prompt, JSON schema reply, fail-loud)
  → Grounding validation (independent per-claim verification of the ACTUAL draft
    text against retrieved evidence — backend/app/generation/claims.py)
  → Multi-signal escalation policy (named reason codes, hard gates for high-risk
    intents, ambiguity/novelty/privacy signals — backend/app/escalation/policy.py)
  → AUTO-HANDLE or ESCALATE
```

Backend: Flask + gunicorn (`backend/app/api/app.py`, `backend/wsgi.py`), request IDs on
every request, one error shape, structured PII-free decision logging, in-memory rate
limit on agent endpoints. All stages are separate modules under `backend/app/`
(`classification/`, `retrieval/`, `generation/`, `escalation/`, `services/`,
`providers/`, `evaluation/`) orchestrated by `services/agent.py.SupportAgent`.

Frontend: static HTML/CSS/JS, no build step, served by the backend (and by nginx in
Docker). Hash-based SPA routing in `frontend/js/app.js`; one JS file per page in
`frontend/js/pages/`; shared DOM helpers in `frontend/js/ui.js`; API client in
`frontend/js/api.js`. Design system is plain CSS custom properties in
`frontend/css/theme.css` (white Linear/Intercom-style theme — already close to the
target aesthetic).

Data: `data/processed/conversations_AmazonHelp.jsonl` (60k reconstructed conversations,
real AmazonHelp Twitter threads with message-level roles and timestamps),
`data/processed/labeled_AmazonHelp.jsonl` (44.4k labeled conversations with intent +
split), `data/golden/` (200 human-verified golden examples + predictions export),
`data/raw/twcs.csv` (Kaggle source). Models in `models/` (classifier, retriever,
evidence weights, intents cfg, retrieval tuning).

## 2. Existing frontend structure

| File | Role |
|---|---|
| `frontend/index.html` | Shell: sidebar + topbar + page root. Title/brand: "Omniroute" |
| `frontend/css/theme.css` | Design tokens + primitives (card, badge, btn, meter, states) |
| `frontend/css/layout.css` | Shell layout, sidebar collapse, responsive rules |
| `frontend/js/app.js` | Hash router, mode pill (provider state machine), keyboard shortcuts |
| `frontend/js/api.js` | Fetch client, single error shape |
| `frontend/js/ui.js` | el(), badges, meters, stat blocks, loading/error/empty boxes |
| `frontend/js/pages/overview.js` | Trust dashboard (gap KPI, provider status) |
| `frontend/js/pages/agent.js` | "Live Agent": composer + pipeline timeline + decision/evidence cards |
| `frontend/js/pages/evaluation.js` | Silver vs golden benchmark panels |
| `frontend/js/pages/failures.js` | Failure analysis (real misclassifications) |
| `frontend/js/pages/golden.js` | Golden set browser |
| `frontend/js/pages/judge.js` | LLM judge status (NOT VALIDATED labeling) |
| `frontend/js/pages/decisions.js` | Design decision log (docs/decision-log.md) |
| `frontend/js/pages/system.js` | System health components |

## 3. Existing backend structure

- `app/api/app.py` — all HTTP endpoints (see §4)
- `app/services/agent.py` — SupportAgent orchestrator (stages, fail-safe, latency)
- `app/services/conversation.py` — thread reconstruction from the tweet graph
- `app/services/ambiguity.py`, `novelty.py`, `evidence_scoring.py`, `text_cleaning.py`,
  `lang_id.py`, `data_split.py`, `ingestion.py`
- `app/classification/baselines.py` — TF-IDF + LogReg classifier
- `app/retrieval/` — retriever (hybrid reranking), lexical, embeddings, quality, vector_store
- `app/generation/generator.py` + `claims.py` — grounded generation + claim verification
- `app/escalation/policy.py` — decision policy + reason codes
- `app/providers/llm/` — groq / gemini / mock providers, factory, measured status
- `app/evaluation/` — metrics, judge, agreement
- `backend/tests/` — unit + integration suites (mock-forced, offline)

## 4. Existing APIs

```
GET  /health                          GET  /api/v1/provider/health?verify=1
GET  /api/v1/intents                  POST /api/v1/agent/classify
POST /api/v1/agent/retrieve           POST /api/v1/agent/respond
GET  /api/v1/evaluation/summary       GET  /api/v1/evaluation/failures
GET  /api/v1/evaluation/automation    GET  /api/v1/evaluation/gap
GET  /api/v1/evaluation/intents       GET  /api/v1/golden-set/summary
GET  /api/v1/golden-set/examples      GET  /api/v1/llm-judge/summary
GET  /api/v1/decisions                GET  /api/v1/system
GET  /<frontend>                      (static SPA)
```

## 5. Existing reusable components

Frontend primitives already exist and are reused (not duplicated): card, badge,
mono-badge, btn, meter/confidenceMeter, statBlock, hbar lists, chips, skeleton,
state boxes (loading/error/empty), callouts, tooltips (data-tip), tables. The mode
pill implements the full provider state machine (MOCK/LIVE/ERROR/NOT_CONFIGURED/
UNVERIFIED/OFFLINE) — preserved as-is.

## 6. Existing evaluation outputs (reports/)

baseline_results, retrieval_metrics, automation_curve (+csv), golden_automation_check,
golden_set_evaluation, leakage_analysis, evidence_score_calibration, failure_analysis,
intent_clusters, judge_human_agreement, judge_live_judge_run, live-groq-validation,
brand_profile, dataset_build_report, golden_gap_analysis, misleading_headline_number,
final_report. Golden set summary in `data/golden/`. All served through
`/api/v1/evaluation/*` — the frontend never hardcodes numbers.

## 7. Existing data flow

UI → `frontend/js/api.js` → Flask endpoints → `SupportAgent.respond()` (single call
returns the full decision trail: intent, evidence, ambiguity, novelty, response,
claim verification, decision, latency) or read-only projections of `reports/*.json`.
Provider truth flows from `/health` (config) + `/api/v1/provider/health?verify=1`
(measured) — the UI only says LIVE after a measured success.

## 8. Existing problems

1. **Product framing** — the shell reads as an ML dashboard: pages are organized around
   evaluation artifacts ("Evaluation", "Golden Set", "LLM Judge" as top-level nav); the
   landing page is a trust/metrics dashboard; there is no inbox, no customer
   conversation, no escalation queue, no resolved view. The support workflow the
   assignment describes is invisible.
2. **No inbox product surface** — `POST /api/v1/agent/respond` exists but is framed as
   "Live Agent" experiment, not as agent-assist inside a support workspace.
3. **No operational queues** — nothing materializes the escalation policy's output
   (ESCALATE decisions) into a workable queue; no resolved-conversation list.
4. **Decision log confusion** — `/api/v1/decisions` returns the DESIGN decision log
   (docs/decision-log.md), not runtime agent decisions. Runtime decisions exist only as
   stdout log lines. There is no per-request traceable store.
5. **Naming** — visible product name is "Omniroute"; assignment calls for a
   support-product identity.
6. Retrieval is visible only after manually submitting a message; there is no
   retrieval-first explorable surface for recruiters.

## 9. What will be preserved (unchanged behavior)

- The entire ML pipeline: classification, retrieval (+hybrid reranking), evidence
  scoring, grounded generation, claim verification, escalation policy.
- All existing endpoints and their response shapes (frontend adapts; API does not break).
- Fail-loud provider semantics, mock/live labeling, provenance metadata.
- All evaluation reports and their generation scripts; golden set.
- Tests; Flask+gunicorn; Docker layout; design-token approach in CSS.
- The mode-pill provider state machine.

## 10. What will be redesigned

- **Product identity:** Omniroute → **EvidenceDesk** ("AI Customer Support Operations")
  across title, sidebar, page titles, README. Internal package/module names, env vars,
  Docker service names, and API paths stay unchanged (safe-renaming rule).
- **Information architecture** (new sidebar):
  - SUPPORT: Inbox · AI Assistant · Escalated · Resolved
  - INSIGHTS: Overview · AI Performance · Retrieval Quality · Failure Analysis
  - EVALUATION: Golden Set · LLM Judge
  - ADMIN: Decision Log · System
- **Inbox (primary page):** three-pane workspace — ticket list (real conversation IDs
  and real messages from the labeled dataset, sanitized display), conversation view
  (real historical thread), AI Copilot panel (intent, calibrated confidence, evidence
  cards with similarity/agreement/relevance, suggested reply, grounding/risk, AUTO/
  ESCALATE decision, expandable "Why?").
- **AI Assistant page:** the existing agent workspace re-framed as paste-a-message →
  full analysis trail (reuses the current Live Agent logic, re-skinned).
- **Escalated queue / Resolved list:** derived from a new runtime decision store.
- **Insights pages:** operational analytics framing; metrics only from the API; silver
  vs human-verified clearly separated.
- **Retrieval Quality:** metrics + Retrieval Explorer (query → top historical cases).
- **Decision Log:** fixed to show runtime agent decisions (from the new store) with the
  design decision log moved to System/About framing.
- **States:** every API-driven section gets loading/error/empty/success states.

## 11. What APIs need modification

- `/api/v1/decisions` — EXTENDED: now returns `{design_decisions, runtime_decisions}`
  (runtime from the new store, newest first, paginated). Kept backward-compatible by
  preserving the previous top-level shape fields.
- `/api/v1/system` — EXTENDED with a `product` block (name, subtitle) so the UI can
  render identity from the backend.

## 12. What new APIs are required

```
GET  /api/v1/inbox                  ticket list for the inbox left pane
                                    (?q=&intent=&status=&page=) — derived from
                                    labeled_AmazonHelp.jsonl, sanitized on the fly
GET  /api/v1/conversations/<id>     full real thread for the middle pane
GET  /api/v1/escalations            runtime ESCALATE decisions (queue)
GET  /api/v1/resolved               runtime AUTO decisions (resolved list)
GET  /api/v1/agent/decisions        runtime decision log (paginated, newest first)
GET  /api/v1/agent/decisions/<id>   one full decision trail
GET  /api/v1/retrieval/explorer     retrieval-only view for the explorer
                                    (?q=&k=) — retrieval metrics stay in
                                    /api/v1/evaluation/summary
```

Runtime decision persistence: a small append-only JSONL store under `data/runtime/`
(directories auto-created; graceful empty-state when absent) written by the existing
decision logging hook. No new dependencies. The store records the request ID, intent,
confidence, evidence score, grounding, decision, reason codes, provider, mode, latency,
and the customer message (already PII-scrubbed display conventions apply).

## 13. Git history rewrite record (attribution cleanup)

Completed 2026-09-16 per the pre-redesign audit's safety protocol:

- **OLD HEAD:** `17d9e418f10231afe13de2e14b5849420929326d`
- **NEW HEAD:** `34606f2d494bac9469017634e000b1f8ff335527`
- **What changed:** commit messages only — the trailer lines
  "🤖 Generated with Codebuff" and "Co-Authored-By: Codebuff" were stripped from 12
  historical commits via `git filter-branch --msg-filter` with
  `scripts/strip_commit_attribution.py`.
- **What did NOT change:** source code, file contents, commit count (32 pre = 32
  post), authors, or dates. Verified with `git diff OLD NEW --stat` (empty) and a
  commit-count comparison.
- **Why:** the assignment prohibits generated-AI attribution in history.
- **Safety:** the working tree was clean; a backup branch
  (`backup/pre-attribution-cleanup`) and tag (`pre-attribution-cleanup-backup`)
  were created at the old HEAD before rewriting. Both are LOCAL ONLY — they contain
  the pre-cleanup messages and must not be pushed; delete them once satisfied:
  `git branch -D backup/pre-attribution-cleanup && git tag -d pre-attribution-cleanup-backup`.
- **Verification:** `git log main --format=%B | grep -i codebuff` returns nothing;
  no repository file contains the attribution strings.
- **Remote note:** the remote `origin/main` still points at pre-rewrite history.
  Because SHAs changed, publishing requires a force-push
  (`git push --force-with-lease origin main`). Coordinate with any collaborators
  before doing so — do not push unless that is expected for this take-home repo.

## 14. Risks and mitigations

- **Dataset size in browser** — inbox API paginates (page size 25) and returns only the
  fields the list needs; conversation payloads are single threads.
- **Never fabricate metrics** — all numbers continue to come from `reports/` via the API;
  unavailable metrics render "Not evaluated"/"Data unavailable".
- **Mock/live clarity** — the provenance block from `/agent/respond` is rendered on
  every suggested reply; mock output is always visibly labeled.
- **History safety** — before any history rewrite: backup branch + original HEAD SHA
  recorded; `git filter-branch --msg-filter` with the existing
  `scripts/strip_commit_attribution.py`; verification greps must return nothing.
