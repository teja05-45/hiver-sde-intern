/* Shared UI helpers. Plain DOM — no framework, no build step. */
(function () {
  "use strict";

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "html") node.innerHTML = v; // only ever called with our own templates
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (k === "tip") node.setAttribute("data-tip", v);
        else node.setAttribute(k, v);
      }
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  const pct = (v, digits = 1) =>
    v === null || v === undefined || Number.isNaN(+v) ? "—" : (v * 100).toFixed(digits) + "%";
  const num = (v, digits = 2) =>
    v === null || v === undefined || Number.isNaN(+v) ? "—" : (+v).toFixed(digits);

  function meter(value, tone) {
    const span = el("span");
    span.style.width = Math.max(0, Math.min(100, (value || 0) * 100)) + "%";
    return el("div", { class: "meter" + (tone ? " " + tone : "") }, span);
  }

  function confidenceMeter(value) {
    const tone = value >= 0.75 ? "green" : value >= 0.5 ? "amber" : "red";
    return meter(value, tone);
  }

  function badge(text, tone = "gray", tip) {
    return el("span", { class: "badge " + tone, text, tip });
  }

  function intentBadge(name) {
    return el("span", { class: "mono-badge", text: name });
  }

  function statBlock({ label, value, sub, tone, tip }) {
    const classes = ["stat"];
    if (tone === "golden") classes.push("golden");
    if (tone === "silver") classes.push("silver");
    return el(
      "div",
      { class: classes.join(" "), tip: tip || null },
      el("div", { class: "label", text: label }),
      el("div", { class: "value", text: value }),
      sub ? el("div", { class: "sub", text: sub }) : null
    );
  }

  function skeletonLines(n = 4, cls = "line") {
    return el(
      "div",
      null,
      Array.from({ length: n }, () => el("div", { class: "skeleton " + cls }))
    );
  }

  function loadingBox(text = "Loading…") {
    return el("div", { class: "state-box", text });
  }

  function errorBox(err) {
    const req = err && err.requestId ? ` · request ${err.requestId.slice(0, 8)}` : "";
    return el(
      "div",
      { class: "state-box error", role: "alert" },
      el("div", { class: "title", text: (err && err.message) || "Something went wrong" }),
      el("div", { class: "small", text: `Code: ${(err && err.code) || "UNKNOWN"}${req}` })
    );
  }

  function emptyBox(title, hint) {
    return el(
      "div",
      { class: "state-box" },
      el("div", { class: "title", text: title }),
      hint ? el("div", { class: "small", text: hint }) : null
    );
  }

  function callout(tone, title, body) {
    return el(
      "div",
      { class: "callout " + tone },
      el("div", { class: "callout-title", text: title }),
      body
    );
  }

  function card(title, bodyChildren, actions) {
    return el(
      "div",
      { class: "card" },
      title
        ? el(
            "div",
            { class: "card-header" },
            el("h2", { text: title }),
            actions || null
          )
        : null,
      el("div", { class: "card-body" }, bodyChildren)
    );
  }

  function kv(key, value) {
    return el(
      "div",
      { class: "row between small" },
      el("span", { class: "text-muted", text: key }),
      el("span", { class: "mono", text: value })
    );
  }

  window.UI = {
    el, pct, num, meter, confidenceMeter, badge, intentBadge, statBlock,
    skeletonLines, loadingBox, errorBox, emptyBox, callout, card, kv,
  };
})();
