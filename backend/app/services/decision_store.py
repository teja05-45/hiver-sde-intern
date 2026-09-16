"""
Runtime decision store: append-only JSONL record of every AUTO/ESCALATE
decision the agent actually made, keyed by request ID.

Purpose: make agent decisions TRACEABLE in the product (Decision Log,
Escalated queue, Resolved list) without adding a database dependency.
One JSON line per decision, appended under a process-wide lock; reads
scan newest-first. The file is git-ignored runtime state, not an
evaluation artifact -- the evaluation source of truth stays in reports/.

Graceful degradation: a missing or unreadable store is an EMPTY queue
with a logged warning, never a 500 -- an empty decision history is a
normal state for a fresh deployment.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_BYTES = 5_000_000  # ~5MB before the oldest half is rotated away


@dataclass
class DecisionRecord:
    request_id: str
    created_at: str            # ISO-8601 UTC
    customer_message: str      # display-sanitized before storing
    intent: str
    intent_confidence: float
    evidence_score: float
    grounding_score: float | None
    decision: str              # "AUTO" | "ESCALATE"
    risk_level: str
    reason: str
    reason_codes: list[str]
    provider: str
    mode: str                  # "mock" | "live"
    latency_ms: int
    conversation_id: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class DecisionStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        try:
            with self._lock:
                self._ensure_parent()
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record.as_dict()) + "\n")
                self._maybe_rotate()
        except OSError as e:
            # Decision capture must never break the support request itself.
            logger.warning("Decision store write failed: %s", e)

    def _maybe_rotate(self) -> None:
        try:
            if not self._path.exists() or self._path.stat().st_size <= MAX_BYTES:
                return
            lines = self._path.read_text(encoding="utf-8").splitlines()
            keep = lines[len(lines) // 2:]
            self._path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            logger.info("Decision store rotated; kept %d newest records", len(keep))
        except OSError as e:
            logger.warning("Decision store rotation failed: %s", e)

    # ------------------------------------------------------------------ reads

    def _read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return [json.loads(ln) for ln in self._path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Decision store unreadable, treating as empty: %s", e)
            return []

    def list_decisions(self, decision: str | None = None,
                       limit: int = 100, offset: int = 0) -> list[dict]:
        records = self._read_all()
        records.reverse()  # newest first
        if decision:
            records = [r for r in records if r.get("decision") == decision]
        return records[offset : offset + max(0, limit)]

    def count_by_decision(self) -> dict[str, int]:
        counts = {"AUTO": 0, "ESCALATE": 0}
        for r in self._read_all():
            d = r.get("decision")
            if d in counts:
                counts[d] += 1
        return counts

    def get(self, request_id: str) -> dict | None:
        return next((r for r in self._read_all() if r.get("request_id") == request_id), None)


def build_record_from_result(result, provenance: dict, display_message: str,
                             conversation_id: str | None = None) -> DecisionRecord | None:
    """Map an AgentResult (+ provenance block from the API layer) to a record.

    Returns None if the result lacks a usable request id -- a decision
    that cannot be traced later is not worth storing.
    """
    request_id = getattr(result, "request_id", None)
    if not request_id:
        return None
    decision = result.decision
    gen = result.generated
    return DecisionRecord(
        request_id=request_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        customer_message=display_message,
        intent=result.intent,
        intent_confidence=round(result.intent_confidence, 4),
        evidence_score=result.evidence_score,
        grounding_score=(round(result.grounding_score, 4) if result.grounding_score is not None else None),
        decision=decision.public_decision,
        risk_level=decision.risk_level.value,
        reason=decision.reason,
        reason_codes=list(decision.reason_codes),
        provider=provenance.get("provider") or "unknown",
        mode=provenance.get("mode") or ("mock" if (gen is not None and gen.is_mock) else "live"),
        latency_ms=int(sum(result.latency_ms.values())) if result.latency_ms else 0,
        conversation_id=conversation_id,
    )
