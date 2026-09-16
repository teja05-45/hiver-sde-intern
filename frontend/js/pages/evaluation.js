/* Evaluation — "How well does it actually perform?" */
(function () {
  "use strict";
  const { el, pct, num, statBlock, skeletonLines, errorBox, card, badge, intentBadge, meter, emptyBox } = UI;

  async function render(root) {
    root.appendChild(skeletonLines(6, "block"));
    let summary, automation, intents;
    const errs = [];
    try { summary = await API.evaluationSummary(); } catch (e) { errs.push(e); }
    try { automation = await API.automation(); } catch (e) { errs.push(e); }
    try { intents = await API.intentMetrics(); } catch (e) { errs.push(e); }
    if (!summary && !automation && !intents) { root.replaceChildren(errorBox(errs[0])); return; }

    const parts = [headSection()];

    const silver = summary?.baselines?.tfidf_logreg?.metrics;
    const majority = summary?.baselines?.majority?.metrics;
    const golden = summary?.golden_set?.results?.tfidf_logreg;
    const goldenMajority = summary?.golden_set?.results?.majority;
    if (silver) parts.push(modelPerformance(silver, golden, majority, goldenMajority));
    if (summary?.retrieval) parts.push(retrievalSection(summary.retrieval));
    if (automation?.curve || automation?.golden_check) parts.push(automationSection(automation));
    if (intents) parts.push(perIntentSection(intents));
    parts.push(silverCaveat(summary?.golden_set?.silver_vs_golden_comparison));

    root.replaceChildren(...parts);
  }

  function headSection() {
    return el("div", { class: "page-head" },
      el("h1", { text: "Evaluation" }),
      el("p", { class: "desc" },
        "All metrics are computed by committed pipeline scripts and served from reports/ via the API — ",
        "nothing on this page is hardcoded. Silver metrics use cluster-derived labels; golden metrics use the ",
        "200-example human-verified set. Read the two together, never separately."));
  }

  function modelPerformance(silver, golden, majority, goldenMajority) {
    return card("Model performance",
      el("div", null,
        el("div", { class: "section-title", text: "Golden set — human-verified labels (the trustworthy number)" }),
        el("div", { class: "grid cols-4" },
          statBlock({ label: "Accuracy", value: pct(golden?.accuracy), sub: `human-verified, n=${golden?.n_examples ?? "—"}`, tone: "golden" }),
          statBlock({ label: "Macro-F1", value: num(golden?.macro_f1), tone: "golden" }),
          statBlock({ label: "Precision (macro)", value: num(golden?.macro_precision), tone: "golden" }),
          statBlock({ label: "Recall (macro)", value: num(golden?.macro_recall), tone: "golden" })),
        el("div", { class: "section-title", text: "Silver benchmark — cluster-derived labels (structurally optimistic)" }),
        el("div", { class: "grid cols-4" },
          statBlock({ label: "Accuracy", value: pct(silver.accuracy), sub: `n=${silver.n_examples}`, tone: "silver", tip: "Labels produced by the same clustering that shaped training — never quote alone." }),
          statBlock({ label: "Macro-F1", value: num(silver.macro_f1), tone: "silver" }),
          statBlock({ label: "Precision (macro)", value: num(silver.macro_precision), tone: "silver" }),
          statBlock({ label: "Recall (macro)", value: num(silver.macro_recall), tone: "silver" })),
        el("div", { class: "mt-5 section-title", text: "Baseline comparison — same metric, both label sources, side by side" }),
        baselineTable(silver, golden, majority, goldenMajority)
      ));
  }

  function baselineTable(silver, golden, majority, goldenMajority) {
    const rows = [
      ["Majority class", pct(majority?.accuracy), num(majority?.macro_f1), pct(goldenMajority?.accuracy), "always predicts the single most frequent intent"],
      ["TF-IDF + LogisticRegression", pct(silver.accuracy), num(silver.macro_f1), pct(golden?.accuracy), "the classifier this system is built on"],
      ["Full agent (retrieval + evidence + grounding + escalation)", "—", "—", "—", "evaluated via automation precision/coverage below — its value is abstention, not intent accuracy"],
    ];
    return el("table", { class: "data" },
      el("thead", null, el("tr", null,
        el("th", { text: "Model" }), el("th", { class: "num", text: "Accuracy (silver)" }),
        el("th", { class: "num", text: "Macro-F1 (silver)" }), el("th", { class: "num", text: "Accuracy (golden)" }),
        el("th", { text: "Notes" }))),
      el("tbody", null, rows.map((r) => el("tr", null,
        el("td", null, el("strong", { text: r[0] })),
        el("td", { class: "num", text: r[1] }), el("td", { class: "num", text: r[2] }),
        el("td", { class: "num", text: r[3] }), el("td", { class: "small text-muted", text: r[4] }))))
    );
  }

  function retrievalSection(r) {
    return card("Retrieval (evidence quality bottleneck)",
      el("div", null,
        el("div", { class: "grid cols-4" },
          statBlock({ label: "Recall@1", value: pct(r.recall_at_k?.["recall@1"]) }),
          statBlock({ label: "Recall@3", value: pct(r.recall_at_k?.["recall@3"]) }),
          statBlock({ label: "Recall@5", value: pct(r.recall_at_k?.["recall@5"]) }),
          statBlock({ label: "MRR", value: num(r.mrr) })),
        el("p", { class: "hint mt-4" },
          `Relevance proxy: retrieved case shares the query's silver intent label (n=${r.n_queries} queries, index=${r.index_size} cases from earlier splits only). `,
          "TF-IDF cannot match paraphrases (\"package late\" vs \"order delayed\"), which caps Recall@1 — the dynamically shown value above is the honest constraint on evidence quality, and sentence embeddings are the identified fix.")));
  }

  function automationSection({ curve, golden_check }) {
    const body = el("div");

    if (curve?.chosen_operating_point) {
      const c = curve.chosen_operating_point;
      body.appendChild(el("div", { class: "grid cols-3" },
        statBlock({ label: "Silver Auto Precision", value: pct(c.auto_precision), sub: `coverage ${pct(c.coverage)} · n_auto=${c.n_auto}`, tip: "Estimated on silver labels at the chosen operating point." }),
        statBlock({ label: "Golden Auto Precision", value: pct(golden_check?.golden_based_measurement?.auto_precision), sub: golden_check?.golden_based_measurement ? `95% CI ${pct(golden_check.golden_based_measurement.wilson_95ci[0],1)}–${pct(golden_check.golden_based_measurement.wilson_95ci[1],1)} · n_auto=${golden_check.golden_based_measurement.n_auto}` : "", tone: "golden" }),
        statBlock({ label: "Golden Coverage", value: pct(golden_check?.golden_based_measurement?.coverage), sub: "same untouched threshold", tone: "golden" })
      ));
    }

    const chartBox = el("div", { class: "mt-5" });
    body.appendChild(chartBox);
    if (curve?.full_curve?.length) {
      renderCoverageChart(chartBox, curve.full_curve, curve.chosen_operating_point, golden_check?.golden_based_measurement);
    } else {
      chartBox.appendChild(emptyBox("No automation curve data", "Run scripts/evaluate_automation.py"));
    }

    body.appendChild(el("div", { class: "chart-legend mt-3" },
      el("span", { class: "key" }, el("span", { class: "swatch", style: "background: var(--accent)" }), "silver curve (cluster-derived)"),
      golden_check?.golden_based_measurement
        ? el("span", { class: "key" }, el("span", { class: "dot-sample", style: "background: var(--green)" }), "golden verified point (human-verified)")
        : null));

    if (golden_check?.conclusion) {
      body.appendChild(el("p", { class: "small text-secondary mt-4", text: golden_check.conclusion }));
    }
    body.appendChild(el("p", { class: "hint mt-3" },
      "Grounding-score contribution to automation quality is NOT AVAILABLE IN MOCK MODE — it requires live LLM generation to measure."));
    return card("Automation — precision vs coverage", body);
  }

  /* SVG precision-vs-coverage chart. X: coverage, Y: precision. */
  function renderCoverageChart(box, points, chosen, golden) {
    const W = 760, H = 320, PAD = { l: 52, r: 16, t: 14, b: 40 };
    const xs = (v) => PAD.l + v * (W - PAD.l - PAD.r);
    const ys = (v) => H - PAD.b - (v - 0.9) / 0.1 * (H - PAD.t - PAD.b); // precision 0.90–1.00

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", "100%");
    svg.style.background = "var(--surface)";

    const mk = (tag, attrs, text) => {
      const n = document.createElementNS(svgNS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (text) n.textContent = text;
      return n;
    };

    // gridlines + axis labels
    for (const p of [0.90, 0.925, 0.95, 0.975, 1.0]) {
      svg.appendChild(mk("line", { x1: PAD.l, x2: W - PAD.r, y1: ys(p), y2: ys(p), stroke: "var(--border)", "stroke-width": 1 }));
      svg.appendChild(mk("text", { x: PAD.l - 8, y: ys(p) + 4, "text-anchor": "end", "font-size": 11, fill: "var(--text-faint)" }, (p * 100).toFixed(1) + "%"));
    }
    for (const c of [0, 0.1, 0.2, 0.3, 0.4, 0.5]) {
      svg.appendChild(mk("text", { x: xs(c), y: H - PAD.b + 18, "text-anchor": "middle", "font-size": 11, fill: "var(--text-faint)" }, (c * 100).toFixed(0) + "%"));
    }
    svg.appendChild(mk("text", { x: W / 2, y: H - 6, "text-anchor": "middle", "font-size": 11, fill: "var(--text-muted)" }, "Coverage"));
    svg.appendChild(mk("text", { x: 14, y: H / 2, "text-anchor": "middle", "font-size": 11, fill: "var(--text-muted)", transform: `rotate(-90 14 ${H / 2})` }, "Auto precision"));

    // silver curve (sorted by coverage)
    const sorted = [...points].sort((a, b) => a.coverage - b.coverage)
      .filter((p) => p.auto_precision >= 0.9);
    const path = sorted.map((p, i) => `${i === 0 ? "M" : "L"}${xs(p.coverage).toFixed(1)},${ys(p.auto_precision).toFixed(1)}`).join(" ");
    svg.appendChild(mk("path", { d: path, fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
    for (const p of sorted) {
      svg.appendChild(mk("circle", { cx: xs(p.coverage), cy: ys(p.auto_precision), r: 2.2, fill: "var(--accent)" }));
    }

    if (chosen) {
      svg.appendChild(mk("circle", { cx: xs(chosen.coverage), cy: ys(chosen.auto_precision), r: 5, fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
    }
    if (golden) {
      svg.appendChild(mk("circle", { cx: xs(golden.coverage), cy: ys(golden.auto_precision), r: 6, fill: "#12b76a" }));
      svg.appendChild(mk("text", { x: xs(golden.coverage) + 10, y: ys(golden.auto_precision) - 8, "font-size": 11, fill: "var(--green)", "font-weight": 600 },
        `golden ${ (golden.auto_precision * 100).toFixed(1) }%`));
    }

    box.appendChild(svg);
  }

  function perIntentSection(intents) {
    const silverRows = intents.silver_per_intent || {};
    const goldenRows = intents.golden_per_intent || {};
    const silverSupport = intents.silver_support || {};
    const goldenSupport = intents.golden_support || {};

    const allIntents = Object.keys(goldenRows).length ? Object.keys(goldenRows) : Object.keys(silverRows);
    const scored = allIntents.map((name) => ({ name, f1: goldenRows[name]?.f1 ?? 0 }));
    const best = scored.reduce((a, b) => (b.f1 > a.f1 ? b : a), { name: "—", f1: -1 });
    const worst = scored.reduce((a, b) => (b.f1 < a.f1 ? b : a), { name: "—", f1: 2 });

    const body = el("div");

    body.appendChild(el("div", { class: "grid cols-3 mb-4" },
      statBlock({ label: "Best intent (golden F1)", value: best.f1 >= 0 ? `${best.name}` : "—", sub: num(best.f1 < 0 ? null : best.f1) }),
      statBlock({ label: "Worst intent (golden F1)", value: worst.name, sub: num(worst.f1 > 1 ? null : worst.f1) }),
      statBlock({ label: "Rare-intent support", value: String(Object.values(goldenSupport).filter((s) => s <= 10).length), sub: "golden intents with ≤10 examples" })));

    body.appendChild(el("div", { class: "section-title", text: "Per-intent F1 — human-verified golden set" }));
    const list = el("div", { class: "hbar-list" });
    for (const name of allIntents) {
      const f1 = goldenRows[name]?.f1 ?? 0;
      const support = goldenSupport[name] ?? silverSupport[name] ?? 0;
      const tone = f1 >= 0.6 ? "var(--accent)" : f1 >= 0.4 ? "#f79009" : "#f04438";
      list.appendChild(el("div", { class: "hbar-row", tip: `support: ${support}` },
        el("span", { class: "name", text: name }),
        el("div", { class: "bar" }, el("span", { style: `width:${(f1 * 100).toFixed(1)}%; background: ${tone}` })),
        el("span", { class: "val", text: num(f1) })));
    }
    body.appendChild(list);

    body.appendChild(el("div", { class: "callout warn mt-5" },
      el("div", { class: "callout-title", text: "About general_other" }),
      el("span", { class: "small text-secondary" },
        "general_other is not a clean semantic intent. It acts partly as an ambiguity/uncertainty bucket — messages ",
        "that are promotional, off-topic, or need context beyond the root message. Treating its metrics like a normal ",
        "business intent overstates the taxonomy's cleanliness; it exists so the system can abstain rather than force a fit." )));

    return card("Per-intent analysis", body);
  }

  function silverCaveat(gap) {
    return el("div", { class: "callout danger mt-6" },
      el("div", { class: "callout-title", text: "Read silver numbers with the golden gap in mind" }),
      el("span", { class: "small text-secondary" },
        gap
          ? `The same classifier scores ${pct(gap.tfidf_logreg_silver_test_accuracy)} on silver labels and ${pct(gap.tfidf_logreg_golden_accuracy)} on human-verified labels — a ${pct(gap.gap)}-point gap caused by label circularity.`
          : "Silver metrics are structurally optimistic; compare against the golden set before drawing conclusions."));
  }

  window.Pages = window.Pages || {};
  window.Pages.evaluation = render;
})();
