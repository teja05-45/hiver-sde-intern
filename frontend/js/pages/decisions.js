/* Decision Log — "Why was the system designed this way?" Rendered from
 * docs/decision-log.md via the API so the UI never duplicates content. */
(function () {
  "use strict";
  const { el, skeletonLines, errorBox, emptyBox } = UI;

  async function render(root) {
    root.appendChild(skeletonLines(5, "block"));
    let data;
    try {
      data = await API.decisions();
    } catch (err) {
      root.replaceChildren(errorBox(err));
      return;
    }

    const entries = data.decisions || [];
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Decision Log" }),
        el("p", { class: "desc" },
          "The non-obvious engineering decisions behind this system, each with its reason and the trade-off accepted. ",
          "Source of truth: docs/decision-log.md.")),
      entries.length
        ? el("div", { class: "card" }, el("div", { class: "card-body" }, entries.map(entryEl)))
        : emptyBox("No decisions found")
    );
  }

  function entryEl(d) {
    // Backend parser normalizes "Tradeoff" / "Trade-off" to key "tradeoff".
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

  window.Pages = window.Pages || {};
  window.Pages.decisions = render;
})();
