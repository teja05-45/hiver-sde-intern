/* Resolved — conversations the AI handled automatically (AUTO decisions).
 * Entries come from the runtime decision store; the resolution source is
 * the AI suggestion path, and each row shows the evidence score and
 * grounding that justified auto-handling. Empty state is honest. */
(function () {
  "use strict";
  const { el, badge, intentBadge, errorBox, emptyBox } = UI;

  async function render(root) {
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Resolved" }),
        el("p", { class: "desc" },
          "Conversations the AI handled without a human. Each entry records the evidence score and grounding ",
          "verification that justified auto-handling — the audit trail behind every automated answer.")),
      el("div", { id: "res-body" }, el("div", { class: "state-box", text: "Loading resolved conversations…" })));

    try {
      const data = await API.resolved();
      renderList(document.getElementById("res-body"), data);
    } catch (err) {
      document.getElementById("res-body").replaceChildren(errorBox(err));
    }
  }

  function renderList(box, data) {
    const items = data.resolved || [];
    if (!items.length) {
      box.replaceChildren(
        emptyBox("No resolved conversations",
          "Nothing has been auto-handled yet. Run an analysis in the Inbox or AI Assistant — decisions that pass every safety gate appear here."));
      return;
    }
    box.replaceChildren(
      el("div", { class: "card" },
        el("div", { class: "card-header" },
          el("h2", { text: `AI-handled (${items.length})` }),
          el("span", { class: "small text-muted", text: "newest first" })),
        el("div", { class: "card-body tight" },
          el("div", { style: "overflow-x: auto;" }, table(items)))));
  }

  function table(items) {
    return el("table", { class: "data" },
      el("thead", null, el("tr", null,
        el("th", { text: "Customer message" }),
        el("th", { text: "Intent" }),
        el("th", { text: "Resolution source" }),
        el("th", { text: "Risk" }),
        el("th", { class: "num", text: "Evidence" }),
        el("th", { class: "num", text: "Grounding" }),
        el("th", { text: "When" }))),
      el("tbody", null, items.map((it) => el("tr", null,
        el("td", { class: "small", style: "max-width: 360px;" },
          el("div", { text: it.customer_message || "(no message stored)" }),
          it.conversation_id ? el("div", { class: "mono small text-muted", text: it.conversation_id }) : null),
        el("td", null, intentBadge(it.intent)),
        el("td", null, badge(`AI · ${it.provider || "—"}`, it.mode === "mock" ? "outline-mock" : "blue",
          it.mode === "mock" ? "Deterministic mock provider — pipeline demonstration, not a live LLM" : "Live LLM provider")),
        el("td", null, badge(it.risk_level || "—",
          it.risk_level === "high" ? "red" : it.risk_level === "medium" ? "amber" : "green")),
        el("td", { class: "num", text: it.evidence_score != null ? Number(it.evidence_score).toFixed(2) : "—" }),
        el("td", { class: "num", text: it.grounding_score != null ? Math.round(it.grounding_score * 100) + "%" : "—" }),
        el("td", { class: "small text-muted nowrap", text: (it.created_at || "").replace("T", " ").slice(0, 16) })))));
  }

  window.Pages = window.Pages || {};
  window.Pages.resolved = render;
})();
