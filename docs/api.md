# API Reference

Base URL: `http://localhost:8000` (backend) or `http://localhost:3000`
(nginx proxy in Docker). All errors use ONE shape:

```json
{"error": {"code": "INVALID_REQUEST", "message": "...", "request_id": "..."}}
```

Every response carries an `X-Request-ID` header (an inbound one is honored)
echoed as `request_id` in bodies and log lines. Rate limiting (120/min by
default, `RATE_LIMIT_PER_MINUTE`) applies to `POST /api/v1/agent/*` only.

## Health & meta

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness + configured mode (`mock_mode` is configuration truth, not measured health) |
| `GET /api/v1/provider/health?verify=1` | Measured provider state: performs a real models-list + 1-token completion. States: `mock`, `configured` (unverified), `healthy`, `unhealthy` with sanitized `error_code` |
| `GET /api/v1/intents` | The 12-intent taxonomy from `configs/intents.yaml` |
| `GET /api/v1/system` | Component health, artifacts (with mtimes), provider status, data-as-of. Never returns secrets |
| `GET /api/v1/decisions` | `docs/decision-log.md` parsed into structured JSON |

## Agent

### `POST /api/v1/agent/classify`
`{"message": str, "k"?}` → `{"request_id", "intent", "confidence", "all_scores"}` —
`all_scores` is the top-5 probability distribution (drives the margin/ambiguity signals).

### `POST /api/v1/agent/retrieve`
`{"message": str, "k": 1..20}` → `{"request_id", "cases": [{conversation_id, similarity, customer_message, resolution, intent}], "intent_agreement_rate"}`

### `POST /api/v1/agent/respond`
The full decision trail. `{"message": str, "k": 1..20}` →

```jsonc
{
  "request_id": "...",
  "customer_message": "...",
  "intent": {
    "name": "delivery_not_received",
    "confidence": 0.99,                    // the classifier's genuine softmax output
    "all_scores": {"delivery_not_received": 0.99, "...": 0.0}
  },
  "evidence": {
    "quality": 0.71,                       // calibrated evidence score
    "features": {"retrieval_score": ..., "intent_agreement_rate": ...,
                  "resolution_consistency": ..., "evidence_count_score": ...},
    "cases": [{"conversation_id", "similarity", "customer_message", "resolution", "intent"}],
    "intent_agreement_rate": 1.0
  },
  "ambiguity": {                           // services/ambiguity.py
    "top2_margin": 0.98, "is_ambiguous": false,
    "multi_intent_suspected": false, "multi_intent_candidates": [],
    "sparse_evidence": false, "retrieval_disagreement": false,
    "noise_suspected": false, "notes": []
  },
  "novelty": {                             // services/novelty.py (OOD signal)
    "ood_score": 0.05, "is_ood": false,
    "subsignals": {"saturated_probability": 0, "low_top2_margin": 0,
                    "low_nearest_similarity": 0, "low_retrieval_agreement": 0,
                    "top_similarity": 0.55, "retrieval_agreement": 1.0},
    "notes": []
  },
  "response": {
    "draft": "...",                        // or null when generation failed/skipped
    "generation_status": "PASS|FAILED|SKIPPED",
    "generation_error_code": null,         // "GENERATION_FAILED" on failure
    "is_mock": false,                      // true => deterministic offline provider
    "grounding_score": 0.83,
    "grounded": true,
    "grounded_claims": [],                 // LLM self-report — audit comparison only
    "unsupported_claims": [],              // LLM self-report — audit comparison only
    "claim_verification": {                // generation/claims.py — DERIVED FROM THE DRAFT
      "claims": [{
        "claim_text": "You can check your latest tracking status.",
        "status": "supported|unsupported|greeting|abstention",
        "evidence_ids": ["AmazonHelp_382779"],
        "confidence": 0.82,
        "explanation": "82% word overlap with the retrieved evidence (best case: ...)"
      }],
      "score": 0.83, "passed": true,
      "n_supported": 5, "n_unsupported": 1
    },
    "provider": "groq", "model": "openai/gpt-oss-120b"
  },
  "provenance": {"provider": "groq", "mode": "live|mock", "model": "...", "timestamp": "..."},
  "decision": {
    "decision": "AUTO|ESCALATE",           // public two-state output
    "internal_decision": "AUTO|REVIEW|ESCALATE",
    "risk_level": "low|medium|high",
    "reason": "human-readable explanation",
    "reason_codes": ["UNSUPPORTED_CLAIMS", "MULTI_INTENT", "OOD_REQUEST", "PRIVACY_RISK",
                      "RETRIEVAL_DISAGREEMENT", "WEAK_EVIDENCE", "LOW_EVIDENCE_SCORE",
                      "LOW_INTENT_CONFIDENCE", "INSUFFICIENT_EVIDENCE_COUNT",
                      "AMBIGUOUS_INTENT", "LOW_GROUNDING_SCORE", "GENERATION_FAILED",
                      "RETRIEVAL_FAILED", "HIGH_RISK_INTENT"]
  },
  "latency_ms": {"classification_ms": 3, "retrieval_ms": 41,
                  "evidence_scoring_ms": 6, "generation_ms": 2280}
}
```

**Contract notes**

- Every `claim_verification.claims[].claim_text` is a literal substring of
  `response.draft` — the UI never displays a claim the draft did not make.
- `decision` is the two-state public output; `internal_decision` distinguishes
  borderline REVIEW cases for the coverage curve.
- A live provider failure NEVER becomes mock output: it surfaces as
  `GENERATION_FAILED` + ESCALATE with `provenance.mode` still `"live"`.

## Evaluation (read-only projections of `reports/`)

| Endpoint | Description |
|---|---|
| `GET /api/v1/evaluation/summary` | Baselines, retrieval, automation curve, golden evaluation, leakage, evidence calibration |
| `GET /api/v1/evaluation/gap` | The silver-vs-golden headline comparison (single source of truth for the Overview KPI) |
| `GET /api/v1/evaluation/failures` | Failure taxonomy from real golden-set misclassifications |
| `GET /api/v1/evaluation/automation` | Precision/coverage curve + golden check of the untouched threshold |
| `GET /api/v1/evaluation/intents` | Per-intent metrics on silver and golden sets |
| `GET /api/v1/golden-set/summary` | Golden-set methodology, distributions, category counts |
| `GET /api/v1/golden-set/examples?outcome=&intent=&q=` | Golden examples joined with model predictions |
| `GET /api/v1/llm-judge/summary` | Judge status (`VALIDATED` requires live judge + REAL human scores; proxy-scored runs are labeled) |
