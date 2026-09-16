/* LLM Judge — "Can I trust the evaluator?" This page is honest by design:
 * the harness exists but has NOT been validated with a live LLM + human
 * comparison. If the backend ever reports VALIDATED, real results render
 * automatically. Until then, this page shows methodology, not results. */
(function () {
  "use strict";
  const { el, num, errorBox, card, badge } = UI;

  async function render(root) {
    let data;
    try {
      data = await API.judgeSummary();
    } catch (err) {
      root.replaceChildren(errorBox(err));
      return;
    }

    const validated = data.status === "VALIDATED";
    // pipeline_validated: the judge ran against a LIVE LLM and produced real
    // agreement statistics. human_validated: those statistics were computed
    // against REAL human scores. An intermediate state exists: live judge vs
    // PROXY human scores (derived programmatically from golden labels) —
    // real numbers, but NOT a human-agreement result, and rendered as such.
    const pipelineValidated = data.pipeline_validated === true;
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "LLM Judge" }),
        el("p", { class: "desc" },
          "An LLM-as-judge can score drafted responses across fixed dimensions so quality is measured, not vibes. ",
          "This page states exactly how far that has gotten — and stops.")),
      el("div", { class: "row mb-4" },
        el("span", { class: `stamp ${validated ? "green" : "red"}`, text: validated ? "VALIDATED" : "NOT VALIDATED" }),
        pipelineValidated && !validated ? badge("LIVE JUDGE — PROXY-HUMAN COMPARISON", "amber") : null,
        data.mock_mode ? badge("MOCK MODE", "amber") : null),

      card("What exists vs. what is verified",
        el("div", null,
          el("ul", { class: "reason-list" },
            el("li", { class: "good" }, el("span", { class: "bullet", text: "✓" }),
              el("span", { text: "Judge harness implemented: 7 dimensions, structured scores, mock-detection — backend/app/evaluation/judge.py." })),
            el("li", { class: "good" }, el("span", { class: "bullet", text: "✓" }),
              el("span", { text: "Agreement statistics implemented and unit-tested: Spearman correlation, weighted Cohen's kappa, exact/adjacent agreement — backend/app/evaluation/agreement.py." })),
            el("li", { class: pipelineValidated ? "good" : "bad" }, el("span", { class: "bullet", text: pipelineValidated ? "✓" : "✗" }),
              el("span", { text: pipelineValidated
                ? `Judge scores produced by a live LLM and compared against reference scores (${data.n_judged}/${data.agreement?.n_examples ?? "—"} examples judged successfully).`
                : "NOT DONE: real LLM judge scores (mock judge outputs are fixed neutral placeholders, deliberately uniform)." })),
            el("li", { class: validated ? "good" : "bad" }, el("span", { class: "bullet", text: validated ? "✓" : "✗" }),
              el("span", { text: validated
                ? "Human vs. judge agreement computed on real human-scored responses."
                : pipelineValidated
                  ? "NOT DONE: the comparison used PROXY human scores derived programmatically from golden labels — NOT real human judgments. No human-agreement claim is made."
                  : "NOT DONE: human comparison. No human agreement number exists for this project, and none is claimed." }))),
          !validated ? el("p", { class: "small text-secondary mt-4", text: data.explanation }) : null)),

      methodologyCard(),
      pipelineValidated ? resultsCard(data, validated) : plannedCard(data)
    );
  }

  function methodologyCard() {
    return card("Judge dimensions",
      el("div", null,
        el("table", { class: "data" },
          el("thead", null, el("tr", null, el("th", { text: "Dimension" }), el("th", { text: "What it measures" }))),
          el("tbody", null, [
            ["correctness", "Does the reply actually address the customer's issue?"],
            ["groundedness", "Are claims supported by the retrieved historical evidence?"],
            ["helpfulness", "Would this reply move the customer toward resolution?"],
            ["completeness", "Does it cover the necessary parts of the answer?"],
            ["actionability", "Does the customer know what happens next?"],
            ["brand_consistency", "Tone/policy consistency with the brand's voice."],
            ["safety", "No harmful, fabricated, or over-promising content."],
          ].map(([d, m]) => el("tr", null,
            el("td", null, el("span", { class: "mono-badge", text: d })),
            el("td", { text: m })))))));
  }

  function plannedCard(data) {
    return card("Validation plan (what would make this page real)",
      el("div", null,
        el("ol", { class: "small text-secondary", style: "padding-left: 20px; display: flex; flex-direction: column; gap: 6px" },
          (data.required_to_validate || []).map((s) => el("li", { text: s }))),
        el("div", { class: "mt-4 section-title", text: "Agreement metrics to report" }),
        el("ul", { class: "small text-secondary", style: "padding-left: 20px" },
          el("li", null, el("strong", { text: "Spearman ρ " }), "— rank correlation between judge and human scores per dimension."),
          el("li", null, el("strong", { text: "Weighted Cohen's κ " }), "— chance-corrected agreement allowing 1-point adjacent tolerance on 1–5 scales."),
          el("li", null, el("strong", { text: "Exact / adjacent agreement " }), "— share of scores identical / within one point.")),
        el("p", { class: "hint mt-4" },
          "Why this matters: an unvalidated judge is just another model with opinions. Until judge scores are shown to ",
          "correlate with human judgment on this exact task, any 'response quality: X/5' number would be decoration — ",
          "so none is shown.")));
  }

  function resultsCard(data, humanValidated) {
    const perDim = data.agreement?.per_dimension || [];
    return card(
      humanValidated ? "Agreement results (live judge vs. human)"
                     : "Agreement results (live judge vs. PROXY human scores)",
      el("div", null,
        !humanValidated ? el("div", { class: "mock-warning mb-3" },
          el("span", { class: "icon", text: "⚠" }),
          el("span", { text:
            "The 'human' side of this comparison is a PROGRAMMATIC PROXY derived from golden labels " +
            "(scripts/score_human_proxy.py), not real human ratings. These agreement numbers prove the " +
            "judge pipeline executes and quantify judge-vs-proxy divergence; they are NOT evidence of " +
            "human agreement and must not be quoted as validation of response quality." })) : null,
        el("p", { class: "small text-secondary mb-4", text: `n=${data.agreement?.n_examples ?? "—"} scored examples · judge: ${data.agreement?.judge_model ?? "—"} via ${data.agreement?.judge_provider ?? "—"}` }),
        el("table", { class: "data" },
          el("thead", null, el("tr", null,
            el("th", { text: "Dimension" }), el("th", { class: "num", text: "Spearman ρ" }),
            el("th", { class: "num", text: "Weighted κ" }), el("th", { class: "num", text: "Exact" }),
            el("th", { class: "num", text: "Adjacent" }))),
          el("tbody", null, perDim.map((d) => el("tr", null,
            el("td", null, el("span", { class: "mono-badge", text: d.dimension })),
            el("td", { class: "num", text: d.spearman_r == null ? "—" : num(d.spearman_r, 3) }),
            el("td", { class: "num", text: num(d.weighted_kappa, 3) }),
            el("td", { class: "num", text: pctExact(d.exact_agreement_rate) }),
            el("td", { class: "num", text: pctExact(d.adjacent_agreement_rate) })))))));
  }

  function pctExact(v) { return v == null ? "—" : (v * 100).toFixed(0) + "%"; }

  window.Pages = window.Pages || {};
  window.Pages.judge = render;
})();
