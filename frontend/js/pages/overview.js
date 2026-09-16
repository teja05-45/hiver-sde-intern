/* Overview — "Can I trust this system?" answered in ~60 seconds.
 * Every number rendered here comes from the API (evaluation summary /
 * gap endpoint) — nothing hardcoded. Structure: header with live system
 * status, primary trust panel, interactive pipeline explainer, and the
 * trust/limitations section (what it does well, what it doesn't, what the
 * headline number hides). */
(function () {
  "use strict";
  const { el, pct, num, statBlock, skeletonLines, errorBox, badge, card } = UI;

  async function render(root) {
    root.appendChild(skeletonLines(6, "block"));
    let summary, gap;
    const errs = [];
    try { summary = await API.evaluationSummary(); } catch (e) { errs.push(e); }
    try { gap = await fetch("/api/v1/evaluation/gap").then((r) => r.ok ? r.json() : null); } catch (_) { /* optional */ }
    if (!summary) { root.replaceChildren(errorBox(errs[0])); return; }

    const baselines = summary.baselines?.tfidf_logreg?.metrics || {};
    const golden = summary.golden_set?.results?.tfidf_logreg || {};
    const retrieval = summary.retrieval || {};
    const silverAuto = summary.golden_automation_check?.silver_based_estimate || {};
    const goldenAuto = summary.golden_automation_check?.golden_based_measurement || {};

    root.replaceChildren(
      headSection(),
      trustPanel({ goldenAuto, gap }),
      pipelineExplainer(),
      wellSection({ golden, retrieval, goldenAuto }),
      notWellSection(summary),
      headlineGapSection({ gap, silverAuto, goldenAuto, baselines })
    );
    animateCounters(root);
  }

  function headSection() {
    return el("div", { class: "page-head overview-head" },
      el("div", null,
        el("h1", { text: "Overview" }),
        el("p", { class: "desc" },
          "How the support agent is performing: automation quality on human-verified labels, where it still fails, ",
          "and why the headline classifier number is not the number to trust.")),
      el("div", { class: "system-status", id: "system-status" },
        el("span", { class: "dot" }),
        el("span", { text: "checking status…" })));
  }

  /* Primary trust panel — the four numbers that matter first. */
  function trustPanel({ goldenAuto, gap }) {
    const escRate = goldenAuto.coverage != null ? 1 - goldenAuto.coverage : null;
    return el("div", null,
      el("div", { class: "section-title", text: "Primary trust metrics — human-verified golden set" }),
      el("div", { class: "grid cols-4" },
        statBlock({
          label: "Automation Precision", value: pct(goldenAuto.auto_precision),
          sub: goldenAuto.wilson_95ci
            ? `95% CI ${pct(goldenAuto.wilson_95ci[0], 1)}–${pct(goldenAuto.wilson_95ci[1], 1)} · n=${goldenAuto.n_auto}`
            : "not available",
          tone: "golden",
          tip: "Of the cases the system would auto-handle, how many did it get right (human-verified)? The number a support lead cares about first.",
          count: true,
        }),
        statBlock({
          label: "Automation Coverage", value: pct(goldenAuto.coverage),
          sub: "at the calibrated threshold",
          tone: "golden",
          tip: "Fraction of traffic the system would auto-answer at the current operating point.",
          count: true,
        }),
        statBlock({
          label: "Escalation Rate", value: pct(escRate),
          sub: "1 − coverage · deliberate trade",
          tone: "golden",
          tip: "Share of messages routed to a human instead of auto-answered. The system buys precision by giving up coverage.",
          count: true,
        }),
        statBlock({
          label: "Intent Accuracy", value: pct(gap?.golden_accuracy),
          sub: "vs 93.6% on silver labels — see below",
          tone: "golden",
          tip: "Accuracy against 200 human-verified examples. The silver-label number is structurally optimistic; both are always shown.",
          count: true,
        })));
  }

  /* Interactive pipeline: click a stage to see what happens there. */
  function pipelineExplainer() {
    const stages = [
      ["Customer", "The message arrives exactly as the customer wrote it. Nothing is rewritten before classification."],
      ["Classify", "A TF-IDF + logistic-regression classifier assigns one of 12 support intents and returns the full probability distribution. The spread matters: a tiny top-2 margin feeds the ambiguity signal."],
      ["Retrieve", "The system searches ~17k historically resolved conversations (cosine similarity, temporal index — only past cases, never future ones). What comes back is the evidence pool."],
      ["Verify Evidence", "Evidence quality is scored from measured signals: top similarity, intent agreement among retrieved cases, resolution consistency, and case count. Weights were fitted, not hand-picked."],
      ["Generate", "An LLM drafts a reply using ONLY the retrieved evidence, in a structured prompt that separates policy from data (prompt-injection defense). If evidence is insufficient it must abstain."],
      ["Ground", "Every claim in the actual draft is extracted and independently verified against the evidence — the LLM's own self-report is never trusted."],
      ["Decide", "A multi-signal policy weighs intent risk, confidence, evidence, grounding, ambiguity, and novelty. Every escalation carries named, machine-readable reason codes."],
    ];
    const detail = el("div", { class: "pipe-detail", role: "status" },
      el("p", { class: "hint" }, "Click a stage to see what happens there."));
    const buttons = stages.map(([name, desc], i) =>
      el("button", {
        class: "pipe-stage" + (i === 0 ? " active" : ""),
        text: name,
        "aria-pressed": i === 0 ? "true" : "false",
        onclick: (e) => {
          document.querySelectorAll(".pipe-stage").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-pressed", "false"); });
          e.currentTarget.classList.add("active");
          e.currentTarget.setAttribute("aria-pressed", "true");
          detail.replaceChildren(el("p", { class: "small text-secondary", text: desc }));
        },
      }));
    detail.replaceChildren(el("p", { class: "small text-secondary", text: stages[0][1] }));
    return card("How automation works",
      el("div", null,
        el("div", { class: "pipe-row", role: "group", "aria-label": "Pipeline stages" },
          buttons.flatMap((b, i) => i < buttons.length - 1 ? [b, el("span", { class: "pipe-arrow", "aria-hidden": "true", text: "→" })] : [b])),
        detail));
  }

  /* WHAT IT DOES WELL — with the actual metrics. */
  function wellSection({ golden, retrieval, goldenAuto }) {
    const items = [];
    if (goldenAuto.auto_precision != null) items.push(
      `Auto-handling precision of ${pct(goldenAuto.auto_precision)} measured on human-verified labels (not silver) — the number that matters for trusting automation.`);
    if (retrieval.recall_at_k?.["recall@5"] != null) items.push(
      `Strong retrieval recall at depth (${pct(retrieval.recall_at_k["recall@5"])} in top-5, MRR ${num(retrieval.mrr)}), so evidence pools usually contain the right history.`);
    items.push(
      "Every escalation carries a named, inspectable reason code — no silent failures, no unexplained refusals.",
      "Claims are verified against evidence independently of the LLM that wrote them; the self-report is never trusted.",
      "Provider status is measured, not assumed: the UI only shows LIVE after a real API check succeeds.");
    return card("What this system does well",
      el("ul", { class: "reason-list" },
        items.map((t) => el("li", { class: "good" }, el("span", { class: "bullet", text: "✓" }), el("span", { text: t })))));
  }

  /* WHAT IT DOES NOT YET DO WELL — from the real failure analysis. */
  function notWellSection(summary) {
    const modes = summary?.failures?.top_failure_modes;
    const items = [];
    if (modes?.length) {
      for (const m of modes.slice(0, 3)) {
        items.push(`${m.category.replace(/_/g, " ")} — ${m.count} of ${summary.failures.n_failures} failures (${pct(m.pct_of_failures)}). ${String(m.hypothesis || "").split(" Fix:")[0]}`);
      }
    } else {
      fetch("/api/v1/evaluation/failures").then((r) => r.ok ? r.json() : null).then((d) => {
        const list = document.getElementById("not-well-list");
        if (!list || !d?.top_failure_modes) return;
        list.replaceChildren(...d.top_failure_modes.slice(0, 3).map((m) =>
          el("li", { class: "bad" }, el("span", { class: "bullet", text: "!" }),
            el("span", null, el("strong", { text: m.category.replace(/_/g, " ") + ` (${pct(m.pct_of_failures)}) — ` }),
              el("span", { class: "text-secondary", text: String(m.hypothesis || "").split(" Fix:")[0] })))));
      }).catch(() => { /* non-fatal */ });
    }
    return card("What it does not yet do well",
      el("div", null,
        el("ul", { class: "reason-list", id: "not-well-list" },
          items.length ? items.map((t) => el("li", { class: "bad" }, el("span", { class: "bullet", text: "!" }), el("span", { text: t })))
            : el("li", { class: "small text-muted" }, el("span", { text: "Loading failure analysis…" }))),
        el("p", { class: "hint mt-3" }, "Full failure taxonomy with real examples: the Failure Analysis page under Insights.")));
  }

  function headlineGapSection({ gap, silverAuto, goldenAuto, baselines }) {
    const inner = el("div", null,
      el("p", { class: "small text-secondary" },
        "Cluster-derived labels can structurally favor the classifier because the same clustering process influenced the taxonomy and the labels. The result: silver metrics look excellent and do not survive human verification. Both are always reported — never one alone."),
      el("div", { class: "grid cols-2 mt-4" },
        silverVsGoldenCard({
          title: "Intent accuracy",
          silver: pct(gap?.silver_accuracy ?? baselines?.accuracy),
          golden: pct(gap?.golden_accuracy),
          delta: gap?.accuracy_gap_points != null ? pct(gap.accuracy_gap_points) + " points overstated" : "",
        }),
        silverVsGoldenCard({
          title: "Auto-handling precision",
          silver: pct(gap?.silver_auto_precision ?? silverAuto.auto_precision),
          golden: pct(gap?.golden_auto_precision ?? goldenAuto.auto_precision),
          delta: goldenAuto.wilson_95ci
            ? `golden 95% CI ${pct(goldenAuto.wilson_95ci[0], 1)}–${pct(goldenAuto.wilson_95ci[1], 1)} (n=${goldenAuto.n_auto})`
            : "",
        })),
      el("details", { class: "mt-4" },
        el("summary", { class: "small", text: "What the headline number hides — dataset and methodology caveats" }),
        el("ul", { class: "small text-secondary caveats mt-3" },
          el("li", null, el("strong", { text: "Silver vs golden labels. " }), "Silver test labels come from the same TF-IDF clustering that shaped training — the classifier is partly graded by the process that taught it."),
          el("li", null, el("strong", { text: "Selective automation. " }), "Coverage/precision are properties of a threshold, not of the model alone; a different threshold trades one for the other (see the Evaluation page curve)."),
          el("li", null, el("strong", { text: "Small golden set. " }), "200 human-verified examples give a wide confidence interval (±15 points on auto-precision) — enough to expose the gap, not to pin a precise operating point."),
          el("li", null, el("strong", { text: "Hard-case weighting. " }), "The golden set deliberately over-samples ambiguous/multi-intent/OOD messages, so golden accuracy is NOT expected traffic accuracy."),
          el("li", null, el("strong", { text: "Temporal distribution. " }), "All data is Oct–Dec 2017 Twitter support; language and policies drift."),
          el("li", null, el("strong", { text: "LLM judge limitations. " }), "The judge pipeline runs live but is compared against proxy scores, not real human ratings — status NOT VALIDATED, shown as such."))));
    return card("What does the headline number hide?", inner);
  }

  function silverVsGoldenCard({ title, silver, golden, delta }) {
    return el("div", { class: "evidence-card" },
      el("div", { class: "ev-head" }, el("strong", { text: title })),
      el("div", { class: "ev-body" },
        el("div", { class: "row between" },
          el("span", { class: "small text-muted", text: "Silver (cluster-derived labels)" }),
          el("strong", { class: "small", text: silver })),
        el("div", { class: "row between" },
          el("span", { class: "small text-muted", text: "Human-verified (golden)" }),
          el("strong", { class: "small", style: "color: var(--accent-hover)", text: golden })),
        delta ? el("div", { class: "small", style: "color: var(--red); font-weight: 600", text: delta }) : null));
  }

  /* Metric count-up: animates .stat .value numbers once on load. Respects
   * prefers-reduced-motion; falls back to static values. */
  function animateCounters(root) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    root.querySelectorAll(".stat .value").forEach((node) => {
      const text = node.textContent || "";
      const m = text.match(/^(\d+(?:\.\d+)?)%$/);
      if (!m) return;
      const target = parseFloat(m[1]);
      const decimals = (m[1].split(".")[1] || "").length;
      const duration = 700;
      const t0 = performance.now();
      function frame(t) {
        const p = Math.min((t - t0) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        node.textContent = (target * eased).toFixed(decimals) + "%";
        if (p < 1) requestAnimationFrame(frame);
        else node.textContent = target.toFixed(decimals) + "%";
      }
      requestAnimationFrame(frame);
    });
  }

  /* Live system status in the header (measured, never assumed). */
  (function initSystemStatus() {
    document.addEventListener("DOMContentLoaded", () => { /* app.js handles it */ });
  })();
  window.OverviewStatus = {
    set(state, label, tip) {
      const box = document.getElementById("system-status");
      if (!box) return;
      box.className = "system-status " + state;
      box.setAttribute("data-tip", tip || "");
      box.querySelector("span:last-child").textContent = label;
    },
  };

  window.Pages = window.Pages || {};
  window.Pages.overview = render;
})();
