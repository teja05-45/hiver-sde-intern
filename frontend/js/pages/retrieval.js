/* Retrieval Quality — is the historical evidence the right evidence?
 * Metrics (Recall@k, MRR) come from reports/retrieval_metrics.json via the
 * API. The Retrieval Explorer runs REAL retrieval-only queries against the
 * same index the agent uses — no generation, no decision, no store writes. */
(function () {
  "use strict";
  const { el, pct, num, statBlock, skeletonLines, errorBox, card, badge, intentBadge, emptyBox } = UI;

  async function render(root) {
    root.appendChild(skeletonLines(4, "block"));
    let summary;
    try {
      summary = await API.evaluationSummary();
    } catch (err) {
      root.replaceChildren(errorBox(err)); return;
    }
    const r = summary?.retrieval;

    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Retrieval Quality" }),
        el("p", { class: "desc" },
          "Every AI suggestion is grounded in historically resolved conversations. This page measures whether retrieval ",
          "finds the RIGHT history — and lets you run the retriever yourself on any message.")),
      r ? metricsSection(r) : el("div", { class: "callout warn mb-4" },
        el("div", { class: "callout-title", text: "Retrieval metrics not evaluated" }),
        el("span", { class: "small text-secondary" },
          "reports/retrieval_metrics.json is missing — run scripts/evaluate_retrieval.py. The explorer below still works.")),
      qualitySection(summary?.evidence_calibration),
      explorerSection());
  }

  function metricsSection(r) {
    const recall = r.recall_at_k || {};
    return card("Measured on the temporal evaluation protocol",
      el("div", null,
        el("div", { class: "grid cols-5" },
          statBlock({ label: "Recall@1", value: pct(recall["recall@1"]), tip: "Share of queries whose best historical case ranks first (relevance proxy: shared silver intent)." }),
          statBlock({ label: "Recall@3", value: pct(recall["recall@3"]) }),
          statBlock({ label: "Recall@5", value: pct(recall["recall@5"]) }),
          statBlock({ label: "Recall@10", value: pct(recall["recall@10"] ?? recall["recall@k10"]) }),
          statBlock({ label: "MRR", value: num(r.mrr), tip: "Mean reciprocal rank of the first relevant case." })),
        el("p", { class: "hint mt-4" },
          `Relevance proxy: retrieved case shares the query's silver intent label (n=${r.n_queries} queries, index=${r.index_size} cases from earlier splits only). `,
          "TF-IDF cannot match paraphrases (\"package late\" vs \"order delayed\"), which caps Recall@1 — the shown value is the honest constraint on evidence quality; sentence embeddings are the identified fix.")));
  }

  function qualitySection(calibration) {
    const feats = calibration?.feature_weights;
    if (!feats || !Object.keys(feats).length) return el("div");
    return card("What makes evidence count",
      el("div", null,
        el("p", { class: "small text-secondary mb-3" },
          "Evidence quality is scored from measured signals with weights fitted on dev data (not hand-picked). ",
          "These are the calibrated weights the evidence scorer uses:"),
        el("div", { class: "hbar-list" },
          Object.entries(feats).map(([k, v]) => el("div", { class: "hbar-row" },
            el("span", { class: "name", text: k }),
            el("div", { class: "bar" }, el("span", { style: `width:${(v * 100).toFixed(1)}%` })),
            el("span", { class: "val", text: num(v) }))))));
  }

  /* ---------------- Retrieval Explorer ---------------- */

  const EXPLORER_EXAMPLES = [
    "Where is my refund?",
    "My package was supposed to arrive yesterday",
    "I want to cancel my order",
    "My account was locked and I cannot log in",
    "The Kindle app will not load my books",
  ];

  function explorerSection() {
    const input = el("textarea", {
      "aria-label": "Customer message to retrieve evidence for",
      placeholder: "Type a customer message to see which historical cases the retriever finds…",
    });
    const kSel = el("select", { "aria-label": "Number of cases", style: "max-width: 90px" },
      ...[3, 5, 8, 10].map((k) => el("option", { value: String(k), text: "top " + k })));
    const results = el("div", { id: "retrieval-results", class: "mt-4" },
      emptyBox("Run a query", "Retrieval runs over ~17k historically resolved AmazonHelp conversations."));

    const runBtn = el("button", { class: "btn primary", type: "button", text: "Retrieve cases" });
    runBtn.addEventListener("click", () => run(input.value, parseInt(kSel.value, 10), results, runBtn));
    input.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(input.value, parseInt(kSel.value, 10), results, runBtn);
    });

    const chips = el("div", { class: "chip-row mt-3" },
      EXPLORER_EXAMPLES.map((m) => el("button", {
        class: "chip", type: "button", text: m.length > 34 ? m.slice(0, 34) + "…" : m,
        onclick: () => { input.value = m; run(m, parseInt(kSel.value, 10), results, runBtn); },
      })));

    return el("div", { class: "mt-6" },
      el("div", { class: "section-title", text: "Retrieval Explorer — try it live" }),
      card("Customer message → top historical cases",
        el("div", null,
          input,
          el("div", { class: "row mt-3" }, runBtn, kSel,
            el("span", { class: "hint", text: "Ctrl + Enter · retrieval only — no generation, no decision" })),
          chips,
          results)));
  }

  async function run(query, k, results, btn) {
    if (!query || !query.trim()) { input?.focus?.(); return; }
    btn.disabled = true;
    results.replaceChildren(el("div", { class: "skeleton block" }));
    try {
      const data = await API.retrievalExplorer(query.trim(), k);
      results.replaceChildren(...explorerResults(data));
    } catch (err) {
      results.replaceChildren(errorBox(err));
    } finally {
      btn.disabled = false;
    }
  }

  function explorerResults(data) {
    const cases = data.cases || [];
    const head = el("div", { class: "row between wrap mb-3" },
      el("span", { class: "small text-muted",
        text: `${cases.length} case(s) · intent agreement ${Math.round((data.intent_agreement_rate ?? 0) * 100)}% · resolution agreement ${Math.round((data.resolution_agreement_rate ?? 0) * 100)}%` }),
      data.hybrid_enabled ? badge("hybrid ranking", "gray", "Semantic + lexical + intent-compatibility + resolution-quality reranking") : badge("plain cosine", "gray"));
    if (!cases.length) return [head, emptyBox("No historical cases retrieved", "The agent would escalate — no evidence, no answer.")];
    return [head, ...cases.map((c) => caseCard(c))];
  }

  function caseCard(c) {
    const comps = c.components && Object.keys(c.components).length
      ? Object.entries(c.components).map(([k, v]) => `${k} ${num(v)}`).join(", ") : null;
    return el("div", { class: "evidence-card" },
      el("div", { class: "ev-head" },
        el("div", { class: "row" },
          el("span", { class: "mono-badge", text: "#" + (c.rank_raw ?? "?") }),
          intentBadge(c.intent),
          el("span", { class: "mono small text-muted", text: c.conversation_id })),
        el("span", { class: "badge blue", text: Math.round((c.similarity || 0) * 100) + "% similar",
          tip: "Cosine similarity between this historical message and the query." })),
      el("div", { class: "ev-body" },
        el("div", { class: "quote" }, el("span", { class: "who", text: "Customer asked" }), c.customer_message),
        c.resolution
          ? el("div", { class: "quote" }, el("span", { class: "who", text: "Resolution" }), c.resolution)
          : el("div", { class: "hint" }, "No recorded resolution."),
        c.explanation || comps
          ? el("details", null,
              el("summary", { class: "small text-muted", text: "Why this ranking?" }),
              el("div", { class: "small text-secondary mt-2" },
                el("span", { text: c.explanation || "" }),
                comps ? el("span", { class: "mono small", text: " " + comps }) : null))
          : null));
  }

  window.Pages = window.Pages || {};
  window.Pages.retrieval = render;
})();
