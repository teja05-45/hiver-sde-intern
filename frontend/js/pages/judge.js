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
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "LLM Judge" }),
        el("p", { class: "desc" },
          "An LLM-as-judge can score drafted responses across fixed dimensions so quality is measured, not vibes. ",
          "This page states exactly how far that has gotten — and stops.")),
      el("div", { class: "row mb-4" },
        el("span", { class: `stamp ${validated ? "green" : "red"}`, text: validated ? "VALIDATED" : "NOT VALIDATED" }),
        data.mock_mode ? badge("MOCK MODE", "amber") : null),

      card("What exists vs. what is verified",
        el("div", null,
          el("ul", { class: "reason-list" },
            el("li", { class: "good" }, el("span", { class: "bullet", text: "✓" }),
              el("span", { text: "Judge harness implemented: 7 dimensions, structured scores, mock-detection — backend/app/evaluation/judge.py." })),
            el("li", { class: "good" }, el("span", { class: "bullet", text: "✓" }),
              el("span", { text: "Agreement statistics implemented and unit-tested: Spearman correlation, weighted Cohen's kappa, exact/adjacent agreement — backend/app/evaluation/agreement.py." })),
            el("li", { class: validated ? "good" : "bad" }, el("span", { class: "bullet", text: validated ? "✓" : "✗" }),
              el("span", { text: validated
                ? "Judge scores produced by a live LLM and compared against human scores."
                : "NOT DONE: real LLM judge scores (mock judge outputs are fixed neutral placeholders, deliberately uniform)." })),
            el("li", { class: validated ? "good" : "bad" }, el("span", { class: "bullet", text: validated ? "✓" : "✗" }),
              el("span", { text: validated
                ? "Human vs. judge agreement computed on scored responses."
                : "NOT DONE: human comparison. No human agreement number exists for this project, and none is claimed." }))),
          !validated ? el("p", { class: "small text-secondary mt-4", text: data.explanation }) : null)),

      methodologyCard(),
      validated ? resultsCard(data) : plannedCard(data)
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

  function resultsCard(data) {
    const perDim = data.agreement?.per_dimension || [];
    return card("Agreement results (live-judge vs. human)",
      el("div", null,
        el("p", { class: "small text-secondary mb-4", text: `n=${data.agreement?.n_examples ?? "—"} scored examples` }),
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
