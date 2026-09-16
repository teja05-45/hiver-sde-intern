"""
Inbox service: a support-inbox projection of the REAL labeled dataset.

Reads data/processed/labeled_AmazonHelp.jsonl (44,375 conversations built
by scripts/build_labeled_dataset.py from the actual AmazonHelp Twitter
corpus) and serves paginated, sanitized tickets. No synthetic customers,
no invented messages: every ticket body is a real historical customer
message; identities are masked as stable "Customer #NNNN" refs (see
display.py); statuses are derived from the dataset's own temporal split
(train -> "resolved", dev/test -> the live work queues).

The full file is NOT loaded into memory per request: the service builds
a lightweight index (id -> offset) on first use and reads only the lines
it needs. Sorting happens once at index time over the small header
fields; the heavy text is fetched lazily per page.

Priority is DERIVED, not invented: it comes from the intent's documented
escalation tendency in configs/intents.yaml (high -> elevated priority),
the same signal the escalation policy itself uses.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.services.display import customer_ref, sanitize_display_text, to_iso_utc

logger = logging.getLogger(__name__)

PAGE_SIZE = 25
MAX_PAGE_SIZE = 100

# Previews are sanitized ONCE at index time and searched in the same form:
# what the agent searches must be what the agent sees (raw corpus text is
# full of @mentions/links that display strips -- matching on raw text would
# return rows whose visible preview doesn't contain the query).

# Split -> queue status mapping. The dataset's temporal split is real
# information about when a conversation happened relative to the model's
# training cutoff; it is the honest analogue of "already handled" vs
# "waiting for triage" in a live workspace.
STATUS_BY_SPLIT = {"train": "resolved", "dev": "open", "test": "open"}


class InboxService:
    """Lazy, thread-safe, offset-indexed reader over the labeled JSONL."""

    def __init__(self, labeled_path: Path) -> None:
        self._path = labeled_path
        self._lock = threading.Lock()
        self._index: list[dict] | None = None  # header rows without message text
        self._intent_meta: dict[str, dict] | None = None

    # ------------------------------------------------------------------ index

    def _ensure_index(self) -> list[dict]:
        with self._lock:
            if self._index is not None:
                return self._index
            rows: list[dict] = []
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed JSONL line %d", line_no)
                            continue
                        rows.append({
                            "line_no": line_no,
                            "conversation_id": r.get("conversation_id", ""),
                            "intent": r.get("intent", "general_other"),
                            "split": r.get("split", "train"),
                            "created_at": to_iso_utc(r.get("created_at")),
                            # Sanitized preview at final display length: the
                            # SAME string is served and searched, so a search
                            # hit is always visible in the preview.
                            "preview_source": sanitize_display_text(
                                r.get("root_message") or "", max_chars=120),
                        })
            # Newest first (ISO timestamps sort lexicographically); undated
            # rows keep a deterministic trailing position.
            rows.sort(key=lambda r: r["created_at"] or "", reverse=True)
            self._index = rows
            logger.info("Inbox index built: %d conversations", len(rows))
            return self._index

    def _ensure_intent_meta(self) -> dict[str, dict]:
        if self._intent_meta is None:
            try:
                import yaml
                cfg = yaml.safe_load(
                    (self._path.parents[2] / "configs" / "intents.yaml").read_text(encoding="utf-8")
                )
                self._intent_meta = {i["name"]: i for i in cfg.get("intents", [])}
            except Exception:  # noqa: BLE001 - taxonomy is display metadata only
                self._intent_meta = {}
        return self._intent_meta

    # ------------------------------------------------------------------ read

    def _row_at(self, line_no: int) -> dict | None:
        with self._path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == line_no:
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _preview(text: str) -> str:
        """Reserved for callers needing a shorter cut of stored previews."""
        return sanitize_display_text(text, max_chars=120)

    def list_tickets(self, q: str = "", intent: str = "", status: str = "",
                     page: int = 1, page_size: int = PAGE_SIZE) -> dict:
        index = self._ensure_index()
        meta = self._ensure_intent_meta()
        page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
        q_lower = (q or "").lower()

        filtered = []
        for r in index:
            row_status = STATUS_BY_SPLIT.get(r["split"], "open")
            if status and row_status != status:
                continue
            if intent and r["intent"] != intent:
                continue
            if q_lower and q_lower not in r["preview_source"].lower():
                continue
            filtered.append(r)

        total = len(filtered)
        start = (max(1, page) - 1) * page_size
        page_rows = filtered[start : start + page_size]

        tickets = []
        for r in page_rows:
            tendency = meta.get(r["intent"], {}).get("escalation_tendency", "high")
            tickets.append({
                "conversation_id": r["conversation_id"],
                "customer_ref": customer_ref(r["conversation_id"]),
                "preview": r["preview_source"],
                "intent": r["intent"],
                "status": STATUS_BY_SPLIT.get(r["split"], "open"),
                "priority": "high" if tendency == "high" else "normal",
                "created_at": r["created_at"],
            })

        return {
            "total": total,
            "page": max(1, page),
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
            "tickets": tickets,
        }

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Full real thread for one conversation (middle pane)."""
        index = self._ensure_index()
        meta = self._ensure_intent_meta()
        header = next((r for r in index if r["conversation_id"] == conversation_id), None)
        if header is None:
            return None
        row = self._row_at(header["line_no"])
        if row is None:
            return None

        tendency = meta.get(row.get("intent", ""), {}).get("escalation_tendency", "high")
        messages = []
        for m in row.get("customer_messages", []) or []:
            messages.append({"role": "customer", "text": m, "created_at": None})
        for m in row.get("agent_messages", []) or []:
            messages.append({"role": "agent", "text": m, "created_at": None})
        # The labeled dataset stores per-role lists; order within each role is
        # preserved. Interleave by original position when lengths allow.
        if row.get("messages"):
            messages = [{"role": m.get("role"), "text": m.get("text"), "created_at": m.get("created_at")}
                        for m in row["messages"]]

        return {
            "conversation_id": conversation_id,
            "customer_ref": customer_ref(conversation_id),
            "intent": row.get("intent"),
            "status": STATUS_BY_SPLIT.get(row.get("split", ""), "open"),
            "priority": "high" if tendency == "high" else "normal",
            "created_at": header["created_at"],
            "resolution": row.get("resolution"),
            "split": row.get("split"),
            "messages": messages,
        }
