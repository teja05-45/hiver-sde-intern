/* Escalated — the queue of conversations the AI routed to humans.
 * Every entry is a REAL stored decision from the runtime decision store
 * (POST /api/v1/agent/respond persists each decision). An empty queue is
 * rendered honestly: "No escalated conversations" — never fabricated. */
(function () {
  "use strict";
  const { el, badge, intentBadge, errorBox, emptyBox } = UI;

  async function render(root) {
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Escalated" }),
        el("p", { class: "desc" },
          "Conversations the AI routed to a human, with the exact reason. The queue is populated by real agent ",
          "decisions — analyze messages in the Inbox or AI Assistant and escalations appear here.")),
      el("div", { id: "esc-body" }, el("div", { class: "state-box", text: "Loading escalation queue…" })));

    try {
      const data = await API.escalations();
      renderQueue(document.getElementById("esc-body"), data);
    } catch (err) {
      document.getElementById("esc-body").replaceChildren(errorBox(err));
    }
  }

  function renderQueue(box, data) {
    const items = data.escalations || [];
    if (!items.length) {
      box.replaceChildren(
        emptyBox("No escalated conversations",
          "Nothing has been escalated yet. Run an analysis in the Inbox or AI Assistant — anything the AI cannot safely answer appears here."));
      return;
    }
    box.replaceChildren(
      el("div", { class: "card" },
        el("div", { class: "card-header" },
          el("h2", { text: `Escalation queue (${items.length})` }),
          el("span", { class: "small text-muted", text: "newest first" })),
        el("div", { class: "card-body tight" },
          el("div", { style: "overflow-x: auto;" }, table(items)))));
  }

  function table(items) {
    return el("table", { class: "data" },
      el("thead", null, el("tr", null,
        el("th", { text: "Customer message" }),
        el("th", { text: "Intent" }),
        el("th", { text: "Escalation reason" }),
        el("th", { text: "Risk" }),
        el("th", { class: "num", text: "Evidence" }),
        el("th", { text: "When" }),
        el("th", { text: "" }))),
      el("tbody", null, items.map((it) => {
        const tr = el("tr", null,
          el("td", { class: "small", style: "max-width: 320px;" },
            el("div", { text: it.customer_message || "(no message stored)" }),
            it.conversation_id ? el("div", { class: "mono small text-muted", text: it.conversation_id }) : null),
          el("td", null, intentBadge(it.intent)),
          el("td", null,
            el("div", { class: "row wrap" }, (it.reason_codes || []).map((c) => el("span", { class: "rc-reason", text: c })))),
          el("td", null, badge(it.risk_level || "—",
            it.risk_level === "high" ? "red" : it.risk_level === "medium" ? "amber" : "green")),
          el("td", { class: "num", text: it.evidence_score != null ? Number(it.evidence_score).toFixed(2) : "—" }),
          el("td", { class: "small text-muted nowrap", text: (it.created_at || "").replace("T", " ").slice(0, 16) }),
          el("td", null, el("button", { class: "btn small", type: "button", text: "Inspect", "data-rid": it.request_id })));
        tr.querySelector("button").addEventListener("click", () => openInspector(it));
        return tr;
      })));
  }

  /* Inspector drawer: the full decision record for one escalation. */
  function openInspector(record) {
    document.body.classList.add("drawer-open");
    const backdrop = el("div", { class: "drawer-backdrop", onclick: close });
    const drawer = el("div", { class: "drawer open", role: "dialog", "aria-modal": "true", "aria-label": "Escalation details" },
      el("button", { class: "btn small drawer-close", type: "button", text: "Close", onclick: close }),
      el("div", { class: "copilot-section", text: "Escalation details" }),
      el("h2", { text: (record.reason_codes || []).join(", ") || "ESCALATED" }),
      el("p", { class: "small text-secondary", text: record.reason || "" }),
      section("Customer message", record.customer_message || "—"),
      section("Detected intent", `${record.intent} · ${Math.round((record.intent_confidence || 0) * 100)}% confidence`),
      section("Evidence", `score ${record.evidence_score != null ? Number(record.evidence_score).toFixed(2) : "—"} · grounding ${
        record.grounding_score != null ? Math.round(record.grounding_score * 100) + "%" : "not run"}`),
      section("Risk", record.risk_level || "—"),
      section("Provider", `${record.provider || "—"} (${record.mode || "—"} mode)`),
      section("Recorded at", record.created_at || "—"),
      record.request_id ? section("Request ID", record.request_id) : null);
    document.body.append(backdrop, drawer);
    function close() {
      document.body.classList.remove("drawer-open");
      backdrop.remove();
      drawer.remove();
    }
  }

  function section(label, value) {
    return el("div", null,
      el("div", { class: "copilot-section", text: label }),
      el("p", { class: "small text-secondary", style: "white-space: pre-wrap; word-break: break-word;", text: String(value) }));
  }

  window.Pages = window.Pages || {};
  window.Pages.escalations = render;
})();
