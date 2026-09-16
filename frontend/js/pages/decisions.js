/* Decision Log — every agent decision, traceable.
 * Primary: RUNTIME decisions actually made by the agent (from the runtime
 * decision store, via /api/v1/agent/decisions) — click a row for the full
 * record. Secondary: the DESIGN decision log (docs/decision-log.md via
 * /api/v1/decisions), the engineering rationale behind the system. */
(function () {
  "use strict";
  const { el, badge, intentBadge, skeletonLines, errorBox, emptyBox } = UI;

  async function render(root) {
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Decision Log" }),
        el("p", { class: "desc" },
          "Every decision the agent actually made — intent, confidence, evidence, risk, decision, reason codes, provider, ",
          "and latency. Click a row for the full record. Engineering design rationale is below.")),
      el("div", { id: "runtime-log" }, el("div", { class: "state-box", text: "Loading decision log…" })),
      el("div", { id: "design-log" }, el("div", { class: "state-box", text: "Loading design decisions…" })));

    loadRuntime();
    loadDesign();
  }

  /* ---------------- runtime decisions ---------------- */

  async function loadRuntime() {
    const box = document.getElementById("runtime-log");
    let data;
    try {
      data = await API.runtimeDecisions({ limit: 200 });
    } catch (err) {
      box.replaceChildren(errorBox(err));
      return;
    }
    const items = data.decisions || [];
    if (!items.length) {
      box.replaceChildren(card(
        emptyBox("No agent decisions recorded yet",
          "Decisions are recorded every time the AI analyzes a message — try the Inbox or AI Assistant. This log is never pre-populated with fake entries.")));
      return;
    }
    box.replaceChildren(card(
      el("div", { style: "overflow-x: auto;" }, runtimeTable(items))));
  }

  function runtimeTable(items) {
    return el("table", { class: "data" },
      el("thead", null, el("tr", null,
        el("th", { text: "Time" }),
        el("th", { text: "Request ID" }),
        el("th", { text: "Intent" }),
        el("th", { class: "num", text: "Conf." }),
        el("th", { class: "num", text: "Evidence" }),
        el("th", { text: "Risk" }),
        el("th", { text: "Decision" }),
        el("th", { text: "Reason codes" }),
        el("th", { text: "Provider" }),
        el("th", { class: "num", text: "Latency" }))),
      el("tbody", null, items.map((it) => {
        const tr = el("tr", { style: "cursor: pointer;", tabindex: "0", role: "button",
          "aria-label": `Inspect decision ${it.request_id}` },
          el("td", { class: "small text-muted nowrap", text: (it.created_at || "").replace("T", " ").slice(0, 19) }),
          el("td", { class: "mono small", text: (it.request_id || "").slice(0, 8) + "…" }),
          el("td", null, intentBadge(it.intent)),
          el("td", { class: "num", text: it.intent_confidence != null ? Math.round(it.intent_confidence * 100) + "%" : "—" }),
          el("td", { class: "num", text: it.evidence_score != null ? Number(it.evidence_score).toFixed(2) : "—" }),
          el("td", null, badge(it.risk_level || "—",
            it.risk_level === "high" ? "red" : it.risk_level === "medium" ? "amber" : "green")),
          el("td", null, badge(it.decision, it.decision === "AUTO" ? "green" : "amber")),
          el("td", null, el("div", { class: "row wrap" },
            (it.reason_codes || []).map((c) => el("span", { class: "rc-reason", text: c })))),
          el("td", { class: "small" }, badge(`${it.provider || "—"}`, it.mode === "mock" ? "outline-mock" : "blue")),
          el("td", { class: "num", text: it.latency_ms != null ? it.latency_ms + " ms" : "—" }));
        const open = () => openInspector(it);
        tr.addEventListener("click", open);
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
        return tr;
      })));
  }

  function openInspector(record) {
    document.body.classList.add("drawer-open");
    const backdrop = el("div", { class: "drawer-backdrop", onclick: close });
    const drawer = el("div", { class: "drawer open", role: "dialog", "aria-modal": "true", "aria-label": "Decision details" },
      el("button", { class: "btn small drawer-close", type: "button", text: "Close", onclick: close }),
      el("div", { class: "copilot-section", text: "Decision record" }),
      el("div", { class: "row" },
        badge(record.decision, record.decision === "AUTO" ? "green" : "amber"),
        badge(`risk: ${record.risk_level || "—"}`,
          record.risk_level === "high" ? "red" : record.risk_level === "medium" ? "amber" : "green")),
      section("Reason", record.reason || "—"),
      section("Customer message", record.customer_message || "(not stored)"),
      section("Intent", `${record.intent} · ${Math.round((record.intent_confidence || 0) * 100)}% confidence`),
      section("Evidence score", record.evidence_score != null ? Number(record.evidence_score).toFixed(4) : "—"),
      section("Grounding score", record.grounding_score != null ? Math.round(record.grounding_score * 100) + "%" : "not run"),
      section("Reason codes", (record.reason_codes || []).join(", ") || "none — all gates passed"),
      section("Provider", `${record.provider || "—"} (${record.mode || "—"} mode)`),
      section("Latency", record.latency_ms != null ? record.latency_ms + " ms total" : "—"),
      record.conversation_id ? section("Conversation", record.conversation_id) : null,
      section("Recorded at", record.created_at || "—"),
      section("Request ID", record.request_id || "—"));
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

  /* ---------------- design decisions (secondary) ---------------- */

  async function loadDesign() {
    const box = document.getElementById("design-log");
    let data;
    try {
      data = await API.decisions();
    } catch (err) {
      box.replaceChildren(errorBox(err));
      return;
    }
    const entries = data.decisions || [];
    box.replaceChildren(
      el("details", { class: "mt-6" },
        el("summary", { class: "section-title", style: "cursor: pointer;", text: `Design decisions — why the system is built this way (${entries.length})` }),
        el("div", { class: "card" },
          el("div", { class: "card-body" },
            el("p", { class: "hint mb-4" }, "Source of truth: docs/decision-log.md — parsed, never duplicated in the frontend."),
            entries.length ? entries.map(entryEl) : emptyBox("No design decisions found")))));
  }

  function entryEl(d) {
    const tradeoff = d.tradeoff || d.trade_off;
    return el("div", { class: "decision-entry" },
      el("div", null, el("span", { class: "num", text: String(d.number) }), el("h3", { text: d.title })),
      el("div", { class: "fields" },
        d.decision ? el("span", { class: "fk", text: "Decision" }) : null,
        d.decision ? el("span", { class: "fv", text: d.decision }) : null,
        d.reason ? el("span", { class: "fk", text: "Reason" }) : null,
        d.reason ? el("span", { class: "fv", text: d.reason }) : null,
        tradeoff ? el("span", { class: "fk", text: "Trade-off" }) : null,
        tradeoff ? el("span", { class: "fv", text: tradeoff }) : null,
        d.result ? el("span", { class: "fk", text: "Result" }) : null,
        d.result ? el("span", { class: "fv", text: d.result }) : null));
  }

  function card(body) {
    return el("div", { class: "card" }, el("div", { class: "card-body" }, body));
  }

  window.Pages = window.Pages || {};
  window.Pages.decisions = render;
})();
