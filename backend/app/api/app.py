"""
Flask API for the evidence-first support agent.

Note on framework choice: the assignment's suggested stack is FastAPI +
Pydantic. This backend is implemented in Flask with hand-written
request/response validation -- it is the version actually executed and
tested here (see backend/tests/). A FastAPI/Pydantic port is a mechanical
rewrite of this same logic; each endpoint's docstring describes its schema
precisely enough to do that directly. See docs/decision-log.md (#13) for
the full reasoning.

Cross-cutting behavior:
  - Every request gets a request ID (X-Request-ID header, echoed in every
    JSON body and in log lines) so a UI user and a log reader can correlate
    an exact backend decision with its structured log record.
  - All errors use ONE shape:
        {"error": {"code": ..., "message": ..., "request_id": ...}}
    Codes: INVALID_REQUEST, NOT_FOUND, INTERNAL_ERROR.
  - Structured logs: one log line per agent request with request_id,
    latency, intent, confidence, evidence score, grounding score, decision,
    and reason codes. Never logs message bodies, API keys, or secrets.
  - CORS: origins come from CORS_ORIGINS (comma-separated). "*" is only
    allowed when APP_ENV=development; production requires explicit origins.

Endpoints:
    GET  /health
    GET  /api/v1/intents
    POST /api/v1/agent/classify            {"message", "k"?}
    POST /api/v1/agent/retrieve            {"message", "k"?}
    POST /api/v1/agent/respond             {"message", "k"?}
    GET  /api/v1/evaluation/summary
    GET  /api/v1/evaluation/failures
    GET  /api/v1/evaluation/automation
    GET  /api/v1/evaluation/gap           (silver vs. human-verified headline gap)
    GET  /api/v1/evaluation/intents
    GET  /api/v1/golden-set/summary
    GET  /api/v1/golden-set/examples
    GET  /api/v1/llm-judge/summary
    GET  /api/v1/decisions
    GET  /api/v1/system
    GET  /<frontend asset>                 (serves the static dashboard)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.status import get_provider_status
from app.services.agent import SupportAgent
from app.escalation.policy import EscalationThresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("omniroute.api")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIGS_DIR = REPO_ROOT / "configs"
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
FRONTEND_DIR = REPO_ROOT / "frontend"

app = Flask(__name__)

_agent: SupportAgent | None = None
_intents_cfg: dict | None = None

# --------------------------------------------------------------------------
# Rate limiting (simple in-memory sliding window, per client IP).
# Scoped to the expensive inference endpoints only (POST /api/v1/agent/*):
# health checks and read-only report endpoints are exempt so orchestrator
# probes and page loads never trip it. RATE_LIMIT_PER_MINUTE<=0 disables.
# This is deliberately per-process (not Redis-backed) -- right-sized for a
# single-container demo; documented in TECHNICAL_DEBT.md.
# --------------------------------------------------------------------------
try:
    RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
except ValueError:
    RATE_LIMIT_PER_MINUTE = 120

_agent_hits: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _client_key() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def _check_rate_limit() -> int:
    """Returns 0 if allowed, otherwise the Retry-After seconds."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return 0
    if not (request.method == "POST" and request.path.startswith("/api/v1/agent/")):
        return 0
    key, now = _client_key(), time.time()
    with _rate_lock:
        hits = _agent_hits[key]
        while hits and now - hits[0] > 60.0:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            return max(int(60.0 - (now - hits[0])) + 1, 1)
        hits.append(now)
    return 0


def _mtime_iso(path: Path) -> str | None:
    """File mtime as ISO-8601 UTC, for cache-invalidation display."""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    except OSError:
        return None


def _data_as_of() -> str | None:
    """Timestamp of the freshest evaluation-report artifact the API serves.
    Lets the UI display "data as of" so stale reports are detectable."""
    mtimes = [_mtime_iso(REPORTS_DIR / f) for f in (
        "baseline_results.json", "golden_set_evaluation.json", "automation_curve.json",
        "failure_analysis.json", "retrieval_metrics.json", "judge_human_agreement.json",
    )]
    mtimes = [m for m in mtimes if m]
    return max(mtimes) if mtimes else None


# --------------------------------------------------------------------------
# Request ID + structured logging + CORS
# --------------------------------------------------------------------------

@app.before_request
def _assign_request_id():
    """Attach a request ID to every request (honors an incoming X-Request-ID
    so upstream gateways can correlate), then enforce the agent-endpoint
    rate limit."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.environ["omniroute_request_id"] = rid
    request.environ["omniroute_start"] = time.perf_counter()

    retry_after = _check_rate_limit()
    if retry_after:
        resp = jsonify({"error": {"code": "RATE_LIMITED",
                                  "message": f"Too many requests. Retry after {retry_after}s.",
                                  "request_id": rid}})
        resp.status_code = 429
        resp.headers["Retry-After"] = str(retry_after)
        return resp


def _request_id() -> str:
    return request.environ.get("omniroute_request_id") or str(uuid.uuid4())


def _latency_ms() -> int:
    start = request.environ.get("omniroute_start")
    return round((time.perf_counter() - start) * 1000) if start else -1


@app.after_request
def _finalize(response):
    rid = _request_id()
    response.headers["X-Request-ID"] = rid
    response.headers["Cache-Control"] = "no-store"

    # CORS from configuration -- never a hardcoded wildcard. "*" is allowed
    # only in development; production requires explicit origins (the origin
    # check itself runs below for credentialed requests).
    settings = get_settings()
    allowed = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    origin = request.headers.get("Origin")
    if origin:
        if "*" in allowed and settings.app_env == "development":
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Request-ID"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

    logger.info(
        json.dumps({
            "event": "http_request",
            "request_id": rid,
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "latency_ms": _latency_ms(),
        })
    )
    return response


@app.route("/<path:_any>", methods=["OPTIONS"], provide_automatic_options=False)
def _cors_preflight(_any):
    return ("", 204)


def api_error(code: str, message: str, status: int = 400):
    """Single error shape for every failure mode."""
    return jsonify({"error": {"code": code, "message": message, "request_id": _request_id()}}), status


@app.errorhandler(404)
def handle_404(_e):
    if request.path.startswith("/api/"):
        return api_error("NOT_FOUND", f"Unknown API path: {request.path}", 404)
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.errorhandler(405)
def handle_405(_e):
    return api_error("INVALID_REQUEST", f"Method {request.method} not allowed on {request.path}", 405)


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.exception("Unhandled exception")
    return api_error("INTERNAL_ERROR", "An unexpected error occurred.", 500)


# --------------------------------------------------------------------------
# Input validation helpers
# --------------------------------------------------------------------------

MAX_MESSAGE_CHARS = 5000
MAX_K = 20


def _validated_message() -> tuple[str, None] | tuple[None, tuple]:
    """Validate the JSON body's `message` field. Returns (message, None) or
    (None, (json_response, status))."""
    if not request.is_json:
        return None, api_error("INVALID_REQUEST", "Request body must be JSON (Content-Type: application/json).", 415)
    body = request.get_json(silent=True)
    if body is None:
        return None, api_error("INVALID_REQUEST", "Request body is not valid JSON.", 400)
    if not isinstance(body, dict):
        return None, api_error("INVALID_REQUEST", "Request body must be a JSON object.", 400)
    message = body.get("message")
    if message is None:
        return None, api_error("INVALID_REQUEST", "'message' is required.", 400)
    if not isinstance(message, str):
        return None, api_error("INVALID_REQUEST", "'message' must be a string.", 400)
    if not message.strip():
        return None, api_error("INVALID_REQUEST", "message must not be empty.", 400)
    if len(message) > MAX_MESSAGE_CHARS:
        return None, api_error("INVALID_REQUEST", f"message exceeds maximum length of {MAX_MESSAGE_CHARS} characters.", 400)
    return message, None


def _validated_k(default: int = 5) -> tuple[int, None] | tuple[None, tuple]:
    body = request.get_json(silent=True) or {}
    k = body.get("k", default)
    if isinstance(k, bool) or not isinstance(k, int):
        return None, api_error("INVALID_REQUEST", "'k' must be an integer.", 400)
    if k < 1 or k > MAX_K:
        return None, api_error("INVALID_REQUEST", f"'k' must be between 1 and {MAX_K}.", 400)
    return k, None


# --------------------------------------------------------------------------
# Agent singleton
# --------------------------------------------------------------------------

def get_agent() -> SupportAgent:
    global _agent, _intents_cfg
    if _agent is None:
        import joblib

        settings = get_settings()
        brand = settings.brand_name or "AmazonHelp"
        classifier = joblib.load(MODELS_DIR / f"classifier_{brand}.joblib")
        retriever = joblib.load(MODELS_DIR / f"retriever_{brand}.joblib")
        weights = json.loads((MODELS_DIR / "evidence_weights.json").read_text())
        intents_cfg = json.loads((MODELS_DIR / "intents_cfg.json").read_text())
        _intents_cfg = intents_cfg
        provider = get_llm_provider(settings)
        _agent = SupportAgent(classifier, retriever, provider, weights, intents_cfg, EscalationThresholds())
    return _agent


def _log_decision(result) -> None:
    """One structured, PII-free log line per agent decision."""
    logger.info(json.dumps({
        "event": "agent_decision",
        "request_id": result.request_id,
        "endpoint": request.path,
        "latency_ms": result.latency_ms,
        "intent": result.intent,
        "intent_confidence": round(result.intent_confidence, 4),
        "retrieval_count": result.evidence.num_cases,
        "evidence_score": result.evidence_score,
        "grounding_score": result.grounding_score,
        "decision": result.decision.public_decision,
        "risk_level": result.decision.risk_level.value,
        "escalation_reason": result.decision.reason,
        "reason_codes": result.decision.reason_codes,
    }))


# --------------------------------------------------------------------------
# Health + meta endpoints
# --------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """GET /health -> {"status", "mock_mode", "llm_provider", "provider_mode",
    "brand", "app_env", "model_loaded", "data_as_of"}

    Configuration truth only: `mock_mode=False` means a live provider is
    CONFIGURED -- it does NOT mean the provider is reachable/healthy. Client
    UI must use /api/v1/provider/health (or /api/v1/system) for measured
    provider state, never infer "live and working" from this endpoint.
    """
    settings = get_settings()
    return jsonify({
        "status": "ok",
        "mock_mode": settings.is_mock_mode(),
        "llm_provider": "mock" if settings.is_mock_mode() else settings.llm_provider,
        "provider_mode": "mock" if settings.is_mock_mode() else "live",
        "brand": settings.brand_name or "AmazonHelp",
        "app_env": settings.app_env,
        "model_loaded": _agent is not None,
        "data_as_of": _data_as_of(),
    })


@app.route("/api/v1/provider/health", methods=["GET"])
def provider_health():
    """GET /api/v1/provider/health?verify=1

    Default: configuration-only status (no external calls). With ?verify=1,
    performs the real minimal provider check (models list + a 1-token
    completion) and reports measured health with latency and a sanitized
    error code. Never returns secrets or raw provider response bodies.
    """
    verify = (request.args.get("verify", "").lower() in ("1", "true", "yes"))
    status = get_provider_status(get_settings(), check_health=verify)
    return jsonify(status)


@app.route("/api/v1/intents", methods=["GET"])
def list_intents():
    """GET /api/v1/intents -> the taxonomy from configs/intents.yaml."""
    intents_path = CONFIGS_DIR / "intents.yaml"
    if not intents_path.exists():
        return api_error("NOT_FOUND", "configs/intents.yaml not found.", 404)
    import yaml
    data = yaml.safe_load(intents_path.read_text(encoding="utf-8"))
    return jsonify(data)


# --------------------------------------------------------------------------
# Agent endpoints
# --------------------------------------------------------------------------

@app.route("/api/v1/agent/classify", methods=["POST"])
def classify():
    """POST /api/v1/agent/classify {"message": str}
    -> {"intent": str, "confidence": float, "all_scores": {intent: float}}"""
    message, err = _validated_message()
    if err:
        return err

    from app.services.text_cleaning import clean_for_modeling
    agent = get_agent()
    pred = agent.classifier.predict([clean_for_modeling(message)])[0]
    return jsonify({
        "request_id": _request_id(),
        "intent": pred.intent,
        "confidence": round(pred.confidence, 4),
        "all_scores": {k: round(v, 4) for k, v in sorted(pred.all_scores.items(), key=lambda kv: -kv[1])[:5]},
    })


@app.route("/api/v1/agent/retrieve", methods=["POST"])
def retrieve():
    """POST /api/v1/agent/retrieve {"message": str, "k": int=5}
    -> {"cases": [...], "intent_agreement_rate": float}"""
    message, err = _validated_message()
    if err:
        return err
    k, err = _validated_k()
    if err:
        return err

    agent = get_agent()
    evidence = agent.retriever.retrieve(message, k=k)
    return jsonify({
        "request_id": _request_id(),
        "cases": [
            {"conversation_id": c.conversation_id, "similarity": round(c.similarity, 4),
             "customer_message": c.customer_message, "resolution": c.resolution, "intent": c.intent}
            for c in evidence.cases
        ],
        "intent_agreement_rate": round(evidence.intent_agreement_rate, 4),
    })


@app.route("/api/v1/agent/respond", methods=["POST"])
def respond():
    """POST /api/v1/agent/respond {"message": str, "k": int=5}
    -> full AgentResult dict (see app.services.agent.AgentResult.as_dict)."""
    message, err = _validated_message()
    if err:
        return err
    k, err = _validated_k()
    if err:
        return err

    agent = get_agent()
    result = agent.respond(message, k=k)
    _log_decision(result)
    payload = result.as_dict()
    payload["request_id"] = _request_id()
    # Response provenance: state which provider ACTUALLY produced the draft
    # (from GeneratedResponse metadata), so mock is never presented as live
    # and a live failure is never relabeled as success.
    settings = get_settings()
    gen = result.generated
    if gen is not None:
        payload["provenance"] = {
            "provider": gen.provider,
            "mode": "mock" if gen.is_mock else "live",
            "model": gen.model or ("mock-deterministic-v1" if gen.is_mock else None),
        }
    else:
        mock = settings.is_mock_mode()
        payload["provenance"] = {
            "provider": "mock" if mock else settings.llm_provider,
            "mode": "mock" if mock else "live",
            "model": "mock-deterministic-v1" if mock else None,
        }
    payload["provenance"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return jsonify(payload)


# --------------------------------------------------------------------------
# Evaluation endpoints (read-only projections of reports/ -- the frontend
# never hardcodes metrics; everything it shows comes from here)
# --------------------------------------------------------------------------

def _read_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read report %s: %s", filename, e)
        return None


@app.route("/api/v1/evaluation/summary", methods=["GET"])
def evaluation_summary():
    """GET /api/v1/evaluation/summary -> silver + golden metrics, retrieval,
    leakage, and the misleading-headline comparison, all from reports/*.json."""
    summary = {}
    for name, filename in [
        ("baselines", "baseline_results.json"),
        ("retrieval", "retrieval_metrics.json"),
        ("automation_curve", "automation_curve.json"),
        ("golden_automation_check", "golden_automation_check.json"),
        ("golden_set", "golden_set_evaluation.json"),
        ("leakage", "leakage_analysis.json"),
        ("evidence_calibration", "evidence_score_calibration.json"),
    ]:
        data = _read_report(filename)
        if data is not None:
            summary[name] = data
    if not summary:
        return api_error("NOT_FOUND", "No evaluation reports found in reports/. Run the pipeline scripts first.", 404)
    return jsonify(summary)


@app.route("/api/v1/evaluation/failures", methods=["GET"])
def evaluation_failures():
    """GET /api/v1/evaluation/failures -> reports/failure_analysis.json"""
    data = _read_report("failure_analysis.json")
    if data is None:
        return api_error("NOT_FOUND", "reports/failure_analysis.json not found. Run scripts/analyze_failures.py first.", 404)
    return jsonify(data)


@app.route("/api/v1/evaluation/automation", methods=["GET"])
def evaluation_automation():
    """GET /api/v1/evaluation/automation -> precision/coverage curve + the
    golden-set check of the silver-calibrated threshold."""
    curve = _read_report("automation_curve.json")
    golden_check = _read_report("golden_automation_check.json")
    if curve is None and golden_check is None:
        return api_error("NOT_FOUND", "Automation reports not found. Run scripts/evaluate_automation.py first.", 404)
    return jsonify({"curve": curve, "golden_check": golden_check})


@app.route("/api/v1/evaluation/gap", methods=["GET"])
def evaluation_gap():
    """GET /api/v1/evaluation/gap -> the ONE source of truth for the silver
    vs. human-verified (golden) headline comparison: accuracy on both label
    sources, the gap, and auto-precision/coverage measured on golden labels.
    The Overview page's primary KPI and 'What does the headline number hide?'
    section render exclusively from this endpoint -- no frontend copy."""
    golden = _read_report("golden_set_evaluation.json")
    golden_check = _read_report("golden_automation_check.json")
    baselines = _read_report("baseline_results.json")
    if golden is None and golden_check is None:
        return api_error(
            "NOT_FOUND",
            "Golden evaluation reports not found. Run scripts/evaluate_against_golden.py "
            "and scripts/evaluate_automation.py first.", 404)
    gap = (golden or {}).get("silver_vs_golden_comparison", {})
    silver_auto = (golden_check or {}).get("silver_based_estimate", {})
    golden_auto = (golden_check or {}).get("golden_based_measurement", {})
    return jsonify({
        "silver_accuracy": gap.get("tfidf_logreg_silver_test_accuracy",
                                   (baselines or {}).get("tfidf_logreg", {}).get("metrics", {}).get("accuracy")),
        "golden_accuracy": gap.get("tfidf_logreg_golden_accuracy"),
        "accuracy_gap_points": gap.get("gap"),
        "silver_auto_precision": silver_auto.get("auto_precision"),
        "silver_auto_coverage": silver_auto.get("coverage"),
        "golden_auto_precision": golden_auto.get("auto_precision"),
        "golden_auto_coverage": golden_auto.get("coverage"),
        "golden_auto_wilson_95ci": golden_auto.get("wilson_95ci"),
        "golden_auto_n": golden_auto.get("n_auto"),
        "golden_n": (golden or {}).get("golden_set_size"),
        "conclusion": (golden_check or {}).get("conclusion"),
    })


@app.route("/api/v1/evaluation/intents", methods=["GET"])
def evaluation_intents():
    """GET /api/v1/evaluation/intents -> per-intent metrics on silver test and
    human-verified golden sets, merged for the per-intent analysis view."""
    baselines = _read_report("baseline_results.json")
    golden = _read_report("golden_set_evaluation.json")
    if baselines is None and golden is None:
        return api_error("NOT_FOUND", "Per-intent reports not found.", 404)
    silver = (baselines or {}).get("tfidf_logreg", {}).get("metrics", {})
    golden_metrics = (golden or {}).get("results", {}).get("tfidf_logreg", {})
    return jsonify({"silver_per_intent": silver.get("per_intent", {}),
                    "golden_per_intent": golden_metrics.get("per_intent", {}),
                    "silver_support": silver.get("support", {}),
                    "golden_support": golden_metrics.get("support", {})})


@app.route("/api/v1/golden-set/summary", methods=["GET"])
def golden_set_summary():
    """GET /api/v1/golden-set/summary -> data/golden/golden_set_summary.json"""
    path = GOLDEN_DIR / "golden_set_summary.json"
    if not path.exists():
        return api_error("NOT_FOUND", "data/golden/golden_set_summary.json not found.", 404)
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/v1/golden-set/examples", methods=["GET"])
def golden_set_examples():
    """GET /api/v1/golden-set/examples -> the golden set joined with the
    model's predictions. Query params (all optional):
        ?outcome=correct|incorrect&intent=<name>&q=<substring>
    Each example: customer message, gold label, model prediction, correct?,
    confidence, categories (ambiguous/multi_intent/noisy/ood/high_risk...)."""
    path = GOLDEN_DIR / "golden_set_with_predictions.json"
    if not path.exists():
        return api_error("NOT_FOUND",
                         "golden_set_with_predictions.json not found. Run scripts/export_golden_with_predictions.py first.", 404)
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data.get("examples", [])

    outcome = request.args.get("outcome")
    intent = request.args.get("intent")
    q = (request.args.get("q") or "").lower()

    def keep(ex):
        if outcome == "correct" and not ex.get("correct"):
            return False
        if outcome == "incorrect" and ex.get("correct"):
            return False
        if intent and ex.get("gold_intent") != intent:
            return False
        if q and q not in ex.get("customer_message", "").lower():
            return False
        return True

    filtered = [ex for ex in examples if keep(ex)]
    return jsonify({"total": len(examples), "returned": len(filtered), "examples": filtered})


@app.route("/api/v1/llm-judge/summary", methods=["GET"])
def llm_judge_summary():
    """GET /api/v1/llm-judge/summary -> judge status + agreement data. In mock
    mode this is explicitly labeled NOT VALIDATED -- the frontend renders it
    as such; this endpoint never presents mock judge numbers as validated."""
    settings = get_settings()
    agreement = _read_report("judge_human_agreement.json")
    is_mock = settings.is_mock_mode()
    mock_flag = bool(agreement.get("is_mock_judge", False)) if agreement else True
    # VALIDATED requires ALL of: live provider (not mock mode), the agreement
    # report explicitly not flagged as a mock judge, actual judged examples
    # (n_judged > 0), and at least one scored dimension. A live provider with
    # zero successful judgments (e.g. all judged calls failed) is explicitly
    # NOT validated -- an empty result must never read as an endorsement.
    n_judged = int(agreement.get("n_judged") or 0) if agreement else 0
    n_dimensions = len((agreement or {}).get("per_dimension") or [])
    validated = (not is_mock) and (not mock_flag) and n_judged > 0 and n_dimensions > 0
    # "Human" scores produced by scripts/score_human_proxy.py are derived from
    # the golden labels programmatically -- they are NOT human ratings. An
    # agreement number computed against them must never be surfaced as a
    # human-validated result.
    human_method = str((agreement or {}).get("human_scoring_method") or "")
    human_labels_real = bool(human_method) and "proxy" not in human_method.lower()
    human_validated = validated and human_labels_real
    return jsonify({
        "status": "VALIDATED" if human_validated else "NOT_VALIDATED",
        "pipeline_validated": validated,
        "human_agreement": "AVAILABLE" if human_validated else "NOT_AVAILABLE",
        "human_scoring_method": human_method or None,
        "mock_mode": is_mock,
        "n_judged": n_judged,
        "explanation": (
            "The LLM-as-judge harness (backend/app/evaluation/judge.py, agreement.py) is fully "
            "implemented and unit-tested, but judge scores in this environment come from the "
            "deterministic mock provider, and no human comparison has been run. These scores are "
            "NOT a real quality assessment and must not be read as one."
            if is_mock else
            (f"A live provider was configured and the judge ran, but {n_judged} examples were "
             f"successfully judged (0 of {agreement.get('n_examples', '?')} attempted "
             "succeeded). No agreement result exists yet -- this is NOT VALIDATED."
             if not validated else
             "Judge scores come from a live LLM and were compared against human scores.")),
        "agreement": agreement,
        "required_to_validate": [
            "Configure LLM_PROVIDER=groq (or gemini) with a real API key.",
            "Generate responses for the golden set with the live provider.",
            "Have a human score ~50 responses on the same dimensions.",
            "Run scripts/compare_judge_to_human.py to compute agreement.",
        ],
    })


@app.route("/api/v1/decisions", methods=["GET"])
def decisions():
    """GET /api/v1/decisions -> docs/decision-log.md rendered as structured
    JSON (decision -> reason -> tradeoff), so the UI has a Decision Log page
    without duplicating its content in the frontend."""
    import re
    doc = REPO_ROOT / "docs" / "decision-log.md"
    if not doc.exists():
        return api_error("NOT_FOUND", "docs/decision-log.md not found.", 404)
    text = doc.read_text(encoding="utf-8")
    entries = []
    blocks = re.split(r"\n(?=### )", text)
    for block in blocks:
        m = re.match(r"### (\d+)\. (.+)", block)
        if not m:
            continue
        num, title = int(m.group(1)), m.group(2).strip()
        body = block[m.end():].strip()
        fields = {}
        # The decision log uses **Decision:** / **Why:** / **Tradeoff:** /
        # **Result:** (also tolerates the older "Reason"/"Trade-off" spellings).
        for key in ("Decision", "Why", "Reason", "Tradeoff", "Trade-off", "Result", "Safety", "Important"):
            fm = re.search(rf"\*\*{re.escape(key)}:\*\*(.+?)(?=\n\*\*|\n###|\Z)", body, re.S)
            if fm:
                # Collapse whitespace but keep the section as a single string.
                val = " ".join(fm.group(1).split())
                if key in ("Why", "Reason"):
                    fields["reason"] = val
                elif key == "Tradeoff" or key == "Trade-off":
                    fields["tradeoff"] = val
                elif key in ("Safety", "Important"):
                    fields[key.lower()] = val
                else:
                    fields[key.lower()] = val
        entries.append({"number": num, "title": title, **fields})
    return jsonify({"decisions": entries})


@app.route("/api/v1/system", methods=["GET"])
def system():
    """GET /api/v1/system -> runtime info the System page shows: provider,
    mode, measured provider health, brand, model artifact presence (with
    mtimes), environment, and a per-component health summary. Never returns
    keys. Provider health here is configuration-derived; the measured
    check lives at /api/v1/provider/health?verify=1."""
    settings = get_settings()
    artifacts = {
        name: {
            "present": (MODELS_DIR / fname).exists(),
            "modified_at": _mtime_iso(MODELS_DIR / fname),
        }
        for name, fname in [
            ("classifier", "classifier_AmazonHelp.joblib"),
            ("retriever", "retriever_AmazonHelp.joblib"),
            ("evidence_weights", "evidence_weights.json"),
            ("intents_cfg", "intents_cfg.json"),
        ]
    }
    mock = settings.is_mock_mode()
    golden_available = (GOLDEN_DIR / "golden_set_summary.json").exists()
    provider = "mock" if mock else settings.llm_provider
    provider_status = get_provider_status(settings, check_health=False)
    healthy = True if mock else provider_status.get("configured", False)
    healthy_label = ("mock (deterministic, offline)" if mock
                     else "configured" if provider_status.get("configured")
                     else "NOT CONFIGURED")
    components = {
        "api": {"status": "healthy", "detail": f"{request.method} serving"},
        "classifier": {"status": "loaded" if artifacts["classifier"]["present"] else "missing",
                       "detail": "classifier_AmazonHelp.joblib"},
        "retriever": {"status": "loaded" if artifacts["retriever"]["present"] else "missing",
                      "detail": "retriever_AmazonHelp.joblib (temporal index)"},
        "evidence_model": {"status": "loaded" if artifacts["evidence_weights"]["present"] else "missing",
                           "detail": "empirically calibrated weights"},
        "llm_provider": {"status": healthy_label,
                         "detail": provider + (" (deterministic, offline)" if mock else
                                   " — run GET /api/v1/provider/health?verify=1 for a measured check" if not mock else "")},
        "golden_set": {"status": "available" if golden_available else "missing",
                       "detail": "200 human-verified examples"},
    }
    return jsonify({
        "brand": settings.brand_name or "AmazonHelp",
        "app_env": settings.app_env,
        "llm_provider": provider,
        "mock_mode": mock,
        "model_name": "mock-deterministic-v1" if mock
                       else (settings.llm_model_name or None),
        "provider_status": provider_status,
        "artifacts_present": artifacts,
        "components": components,
        "api_key_configured": {
            "groq": bool(settings.groq_api_key),
            "gemini": bool(settings.gemini_api_key),
        },
        "data_as_of": _data_as_of(),
        "generated_at": time.time(),
    })


# --------------------------------------------------------------------------
# Frontend (static, no build step)
# --------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@app.route("/<path:filename>", methods=["GET"])
def serve_frontend(filename: str = "index.html"):
    """Serves the static frontend dashboard from /frontend."""
    # The SPA catch-all must not swallow unknown API paths -- those are 404s
    # with the standard error shape, not HTML.
    if filename.startswith("api/") or filename == "health":
        return api_error("NOT_FOUND", f"Unknown API path: /{filename}", 404)
    target = FRONTEND_DIR / filename
    if not target.exists() or not target.is_file():
        filename = "index.html"  # SPA fallback for client-side routes
    # NOTE: pass the full relative subpath (e.g. "js/api.js"), not
    # target.name -- stripping the subdirectory made every asset under
    # css/ or js/ fall through to the index.html fallback.
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    # Development server only. Production deployments must use gunicorn
    # (see Dockerfile.backend / wsgi.py) -- documented in README.
    app.run(host="0.0.0.0", port=8000, debug=False)
