/* Overview — "Can I trust this system?" answered in ~30 seconds.
 * Every number rendered here comes from /api/v1/evaluation/summary. */
(function () {
  "use strict";
  const { el, pct, num, statBlock, skeletonLines, errorBox, callout, badge, card } = UI;

  async function render(root) {
    root.appendChild(skeletonLines(6, "block"));
    let data;
    try {
      data = await API.evaluationSummary();
    } catch (err) {
      root.replaceChildren(errorBox(err));
      return;
    }

    const baselines = data.baselines?.tfidf_logreg?.metrics || {};
    const golden = data.golden_set?.results?.tfidf_logreg || {};
    const retrieval = data.retrieval || {};
    const silverAuto = data.golden_automation_check?.silver_based_estimate || {};
    const goldenAuto = data.golden_automation_check?.golden_based_measurement || {};
    const gap = data.golden_set?.silver_vs_golden_comparison || {};

    root.replaceChildren(
      headSection(),
      kpiSection({ baselines, golden, retrieval, goldenAuto, silverAuto }),
      headlineGapSection({ gap, silverAuto, goldenAuto }),
      explanationSection()
    );
  }

  function headSection() {
    return el(
      "div",
      { class: "page-head" },
      el("h1", { text: "Evidence-First Support Agent" }),
      el(
        "p",
        { class: "desc" },
        "Classifies customer intent, retrieves historically resolved cases, generates evidence-grounded replies, ",
        "validates claims, and escalates when evidence is insufficient. ",
        el("strong", { text: "When evidence is insufficient, it abstains and escalates rather than hallucinate." })
      )
    );
  }

  function kpiSection({ baselines, golden, retrieval, goldenAuto, silverAuto }) {
    return el(
      "div",
      null,
      el("div", { class: "section-title", text: "Human-verified golden set (trustworthy)" }),
      el(
        "div",
        { class: "grid cols-4" },
        statBlock({
          label: "Golden Accuracy", value: pct(golden.accuracy),
          sub: "200 human-verified examples", tone: "golden",
          tip: "Measured against labels verified by hand — not the cluster-derived labels the model was trained on.",
        }),
        statBlock({
          label: "Golden Macro-F1", value: num(golden.macro_f1),
          sub: "per-intent reliability", tone: "golden",
          tip: "Macro-F1 weights rare intents equally; more honest than accuracy under class imbalance.",
        }),
        statBlock({
          label: "Auto Precision", value: pct(goldenAuto.auto_precision),
          sub: goldenAuto.wilson_95ci
            ? `95% CI ${pct(goldenAuto.wilson_95ci[0], 1)}–${pct(goldenAuto.wilson_95ci[1], 1)} · n=${goldenAuto.n_auto}`
            : "",
          tone: "golden",
          tip: "Of the cases the system would auto-handle, how many did it get right (human-verified)?",
        }),
        statBlock({
          label: "Auto Coverage", value: pct(goldenAuto.coverage),
          sub: "at the calibrated threshold", tone: "golden",
          tip: "Fraction of traffic the system would auto-handle at the current operating point.",
        })
      ),
      el("div", { class: "section-title", text: "Silver-label benchmark (cluster-derived labels)" }),
      el(
        "div",
        { class: "grid cols-4" },
        statBlock({
          label: "Silver Accuracy", value: pct(baselines.accuracy),
          sub: `n=${baselines.n_examples ?? "—"} test examples`, tone: "silver",
          tip: "Accuracy against labels produced by the same clustering process that shaped training — structurally optimistic.",
        }),
        statBlock({
          label: "Silver Macro-F1", value: num(baselines.macro_f1), tone: "silver",
        }),
        statBlock({
          label: "Silver Auto Precision", value: pct(silverAuto.auto_precision),
          sub: `coverage ${pct(silverAuto.coverage)}`, tone: "silver",
          tip: "Precision estimated on silver labels — the number that does NOT survive human verification (see below).",
        }),
        statBlock({
          label: "Escalation Rate", value: pct(goldenAuto.coverage ? 1 - goldenAuto.coverage : null),
          sub: "1 − coverage on golden check", tone: "silver",
          tip: "Share of messages routed to a human instead of auto-answered. Deliberately conservative.",
        })
      ),
      el("div", { class: "section-title", text: "Retrieval quality (silver proxy for relevance)" }),
      el(
        "div",
        { class: "grid cols-4" },
        statBlock({ label: "Recall@1", value: pct(retrieval.recall_at_k?.["recall@1"]) }),
        statBlock({ label: "Recall@5", value: pct(retrieval.recall_at_k?.["recall@5"]) }),
        statBlock({ label: "MRR", value: num(retrieval.mrr) }),
        statBlock({ label: "Index Size", value: String(retrieval.index_size ?? "—"), sub: "resolved cases indexed" })
      )
    );
  }

  function headlineGapSection({ gap, silverAuto, goldenAuto }) {
    const inner = el(
      "div",
      null,
      el("p", { class: "small text-secondary" },
        "Cluster-derived labels can structurally favor the classifier because the same clustering process ",
        "influenced the taxonomy and the labels. The result: silver metrics look excellent and do not survive ",
        "human verification."
      ),
      el(
        "div",
        { class: "grid cols-2 mt-4" },
        silverVsGoldenCard({
          title: "Intent accuracy",
          silver: pct(gap.tfidf_logreg_silver_test_accuracy),
          golden: pct(gap.tfidf_logreg_golden_accuracy),
          delta: gap.gap != null ? pct(gap.gap) + " points overstated" : "",
        }),
        silverVsGoldenCard({
          title: "Auto-handling precision",
          silver: pct(silverAuto.auto_precision),
          golden: pct(goldenAuto.auto_precision),
          delta:
            goldenAuto.wilson_95ci
              ? `golden 95% CI ${pct(goldenAuto.wilson_95ci[0], 1)}–${pct(goldenAuto.wilson_95ci[1], 1)} (n=${goldenAuto.n_auto})`
              : "",
        })
      ),
      el("p", { class: "small text-muted mt-4" },
        "Both numbers are always reported, side by side, across this product. The golden numbers are the ones to trust."
      )
    );
    return card("What does the headline number hide?", inner);
  }

  function silverVsGoldenCard({ title, silver, golden, delta }) {
    return el(
      "div",
      { class: "evidence-card" },
      el("div", { class: "ev-head" }, el("strong", { text: title })),
      el(
        "div",
        { class: "ev-body" },
        el("div", { class: "row between" },
          el("span", { class: "small text-muted", text: "Silver (cluster-derived labels)" }),
          el("strong", { class: "small", text: silver })),
        el("div", { class: "row between" },
          el("span", { class: "small text-muted", text: "Human-verified (golden)" }),
          el("strong", { class: "small", style: "color: var(--accent-hover)", text: golden })),
        delta ? el("div", { class: "small", style: "color: var(--red); font-weight: 600", text: delta }) : null
      )
    );
  }

  function explanationSection() {
    return el(
      "div",
      { class: "grid cols-2 mt-6" },
      callout(
        "",
        "Why the gap exists",
        el("span", { class: "small text-secondary" },
          "Training labels were assigned by mapping TF-IDF+KMeans clusters to taxonomy intents. The test set's labels ",
          "come from the same clustering pipeline, so the classifier is partly being graded by the process that taught it. ",
          "The human-verified golden set breaks that circularity — nearly half of its machine-drafted labels had to be ",
          "corrected during manual review (see the Golden Set page for the exact count).")
      ),
      callout(
        "",
        "Why this design is still the right trade",
        el("span", { class: "small text-secondary" },
          "Support automation must optimize safe automation, not raw accuracy. The system deliberately escalates ",
          "high-risk intents (billing, account access), ambiguous messages, and multi-issue complaints — trading ",
          "coverage for precision on what it does auto-handle. Every escalation carries named, inspectable reason codes.")
      )
    );
  }

  window.Pages = window.Pages || {};
  window.Pages.overview = render;
})();
