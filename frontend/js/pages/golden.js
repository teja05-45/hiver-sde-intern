/* Golden Set — "Are the evaluation labels trustworthy?" */
(function () {
  "use strict";
  const { el, pct, num, skeletonLines, errorBox, card, badge, intentBadge, emptyBox, meter } = UI;

  const state = { outcome: "", intent: "", q: "" };

  async function render(root) {
    root.appendChild(skeletonLines(4, "block"));
    let summary;
    try {
      summary = await API.goldenSummary();
    } catch (err) {
      root.replaceChildren(errorBox(err));
      return;
    }

    root.replaceChildren(
      el("div", { class: "page-head" },
      el("h1", { text: "Golden Set" }),
      el("p", { class: "desc" },
        "Human-verified examples — the ground truth that exposed the silver/golden gap. Every message was ",
        "individually reviewed against the taxonomy; the correction count below comes from the backend data.")),
      methodologySection(summary),
      distributionSection(summary),
      browserSection()
    );
  }

  function methodologySection(summary) {
    const cats = summary.category_membership_counts || {};
    const total = summary.total_examples ?? 0;
    const src = summary.label_source_distribution || {};
    const manual = src.manual_primary_labeler_review ?? 0;
    return card("Sampling & labeling methodology",
      el("div", null,
        el("div", { class: "grid cols-4" },
          statBox("Total examples", String(total), "stratified across 8 categories"),
          statBox("Hard cases", String(summary.difficulty_distribution?.hard ?? "—"), `of ${total} — deliberately hard-weighted`),
          statBox("Expected escalation", String(summary.expected_escalation_distribution?.True ?? "—"), "examples expected to escalate"),
          statBox("Labels changed in manual review", String(manual),
            `label provenance: ${Object.entries(src).map(([k, v]) => `${k}=${v}`).join(", ")}`)),
        el("div", { class: "mt-4 section-title", text: "Stratification categories (an example can belong to several)" }),
        el("div", { class: "row wrap" },
          Object.entries(cats).map(([k, v]) => badge(`${k}: ${v}`, "gray"))),
        el("p", { class: "hint mt-4" },
          "Sampled from the temporal test split only (no leakage). Categories come from real pipeline signals: ",
          "classifier top-2 margin < 0.15 → ambiguous; 2+ intents' keyword signals → multi_intent; predicted general_other ",
          "or bottom-decile retrieval similarity → ood. Full methodology: data/golden/README.md.")));
  }

  function distributionSection(summary) {
    const dist = Object.entries(summary.intent_distribution || {}).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...dist.map(([, v]) => v), 1);
    const total = summary.total_examples ?? 0;
    return card("Gold label distribution",
      el("div", { class: "hbar-list" },
        dist.map(([name, count]) =>
          el("div", { class: "hbar-row", tip: `${((count / total) * 100).toFixed(1)}% of the golden set` },
            el("span", { class: "name", text: name }),
            el("div", { class: "bar" }, el("span", { style: `width: ${(count / max * 100).toFixed(1)}%` })),
            el("span", { class: "val", text: String(count) })))));
  }

  function browserSection() {
    const toolbar = el("div", { class: "row wrap mb-4" },
      filterChips(),
      el("select", { id: "gs-intent", style: "max-width: 240px", "aria-label": "Filter by intent" },
        el("option", { value: "", text: "All intents" })),
      el("input", { type: "text", id: "gs-q", placeholder: "Search message text…", style: "max-width: 260px", "aria-label": "Search messages" }));

    const listWrap = el("div", { id: "gs-list" });
    const wrap = el("div", { class: "mt-6" },
      el("div", { class: "section-title", text: "Example browser — gold label vs. model prediction" }),
      toolbar, listWrap);

    loadIntentOptions();
    load(listWrap);

    let debounce;
    wrap.querySelector("#gs-q").addEventListener("input", (e) => {
      clearTimeout(debounce);
      debounce = setTimeout(() => { state.q = e.target.value.trim(); load(listWrap); }, 250);
    });
    wrap.querySelector("#gs-intent").addEventListener("change", (e) => { state.intent = e.target.value; load(listWrap); });
    return wrap;
  }

  function filterChips() {
    const chips = [
      ["", "All"],
      ["correct", "Correct"],
      ["incorrect", "Incorrect"],
      ["highconf", "High confidence"],
      ["lowconf", "Low confidence"],
    ];
    return el("div", { class: "chip-row" },
      chips.map(([val, label]) =>
        el("button", {
          class: "chip" + (state.outcome === val ? " active" : ""),
          text: label,
          onclick: () => {
            state.outcome = val;
            document.querySelectorAll(".chip-row .chip").forEach((c) => c.classList.remove("active"));
            event.target.classList.add("active");
            load(document.getElementById("gs-list"));
          },
        })));
  }

  async function loadIntentOptions() {
    try {
      const data = await API.intents();
      const sel = document.getElementById("gs-intent");
      if (!sel) return;
      for (const intent of data.intents || []) {
        sel.appendChild(el("option", { value: intent.name, text: intent.name }));
      }
    } catch (_) { /* non-critical */ }
  }

  async function load(listWrap) {
    listWrap.replaceChildren(el("div", { class: "skeleton block" }));
    try {
      const params = {};
      if (state.outcome === "correct" || state.outcome === "incorrect") params.outcome = state.outcome;
      if (state.intent) params.intent = state.intent;
      if (state.q) params.q = state.q;
      const data = await API.goldenExamples(params);
      let examples = data.examples || [];

      // confidence filters (computed client-side from API confidence values)
      if (state.outcome === "highconf") examples = examples.filter((e) => e.confidence >= 0.75);
      if (state.outcome === "lowconf") examples = examples.filter((e) => e.confidence < 0.5);

      if (!examples.length) {
        listWrap.replaceChildren(emptyBox("No examples match these filters"));
        return;
      }
      listWrap.replaceChildren(
        el("p", { class: "hint mb-3", text: `${examples.length} of ${data.total} examples shown` }),
        ...examples.slice(0, 50).map(exampleCard),
        examples.length > 50 ? el("p", { class: "hint", text: `…and ${examples.length - 50} more (refine filters to see them).` }) : null
      );
    } catch (err) {
      listWrap.replaceChildren(errorBox(err));
    }
  }

  function exampleCard(ex) {
    const correct = !!ex.correct;
    return el("div", { class: "evidence-card" },
      el("div", { class: "ev-head" },
        el("div", { class: "row" },
          el("span", { class: "mono-badge", text: ex.id }),
          (ex.categories || []).filter((c) => ["ambiguous", "multi_intent", "ood", "noisy", "high_risk", "rare"].includes(c))
            .map((c) => badge(c, c === "ambiguous" || c === "multi_intent" ? "violet" : "amber"))),
        el("span", { class: `verdict ${correct ? "correct" : "incorrect"}`, text: correct ? "✓ Correct" : "✗ Incorrect" })),
      el("div", { class: "ev-body" },
        el("div", { class: "quote", style: "border-left-color: var(--accent-border)" }, ex.customer_message),
        el("div", { class: "row wrap small" },
          el("span", { class: "text-muted", text: "gold:" }), intentBadge(ex.gold_intent),
          el("span", { class: "text-muted", text: "predicted:" }), intentBadge(ex.predicted_intent),
          el("span", { class: "text-muted", text: "confidence:" }), el("strong", { text: num(ex.confidence) })),
        el("div", { style: "max-width: 260px" }, meter(ex.confidence, ex.confidence >= 0.75 ? "green" : ex.confidence >= 0.5 ? "amber" : "red"))));
  }

  function statBox(label, value, sub) {
    return el("div", { class: "stat" },
      el("div", { class: "label", text: label }),
      el("div", { class: "value", text: value }),
      el("div", { class: "sub", text: sub }));
  }

  window.Pages = window.Pages || {};
  window.Pages.golden = render;
})();
