/* Live Agent — "Why did the system make this decision?"
 * Three panes: composer | agent analysis | decision. A pipeline stepper
 * shows every stage the message moved through. */
(function () {
  "use strict";
  const { el, pct, num, badge, intentBadge, meter, confidenceMeter, errorBox, emptyBox, card } = UI;

  const EXAMPLES = [
    "Why hasn't my refund arrived yet?",
    "My package shows delivered but I never received it.",
    "Where is my order? It was supposed to arrive yesterday.",
    "I was charged twice for the same order, this is unacceptable.",
    "My account was locked and I think it was hacked.",
    "The app crashes every time I open my order history.",
  ];

  function render(root) {
    const compose = composePane();
    const analysis = analysisPane();
    const decision = decisionPane();

    root.replaceChildren(
      el(
        "div",
        { class: "page-head" },
        el("h1", { text: "Live Agent" }),
        el("p", { class: "desc" },
          "Submit a customer message and inspect the full decision trail: intent, retrieved historical evidence, ",
          "grounded draft, and the escalation decision with its reason codes.")
      ),
      el("div", { class: "pipeline", id: "pipeline" }, initialPipeline()),
      el(
        "div",
        { class: "workspace mt-4" },
        el("div", { class: "pane pane-compose" }, compose),
        el("div", { class: "pane" }, analysis),
        el("div", { class: "pane" }, decision)
      )
    );

    const textarea = compose.querySelector("textarea");
    compose.querySelector("[data-analyze]").addEventListener("click", () => analyze(textarea.value));
    textarea.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze(textarea.value);
    });
    compose.querySelector("[data-examples]").addEventListener("change", (e) => {
      if (e.target.value) { textarea.value = e.target.value; e.target.value = ""; }
    });

    async function analyze(message) {
      if (!message || !message.trim()) {
        alert("Enter a customer message first.");
        return;
      }
      setPipeline("running");
      const out = document.getElementById("analysis-out");
      const dec = document.getElementById("decision-out");
      out.replaceChildren(el("div", { class: "skeleton block" }), el("div", { class: "skeleton block" }));
      dec.replaceChildren(el("div", { class: "skeleton block" }));

      let result;
      try {
        result = await API.respond(message.trim(), 5);
      } catch (err) {
        out.replaceChildren(errorBox(err));
        dec.replaceChildren(el("div"));
        setPipeline("failed");
        return;
      }
      setPipeline("done", result);
      out.replaceChildren(provenanceCard(result), intentCard(result), evidenceCard(result), generationCard(result));
      dec.replaceChildren(decisionCard(result), groundingCard(result));
    }
  }

  /* ---------- pipeline stepper ---------- */

  const STAGES = [
    ["Intent", (r) => (r ? `${r.intent.name} · ${num(r.intent.confidence)}` : "—"),
      (r) => (r?.latency_ms?.classification_ms != null ? `PASS · ${r.latency_ms.classification_ms} ms` : "PASS")],
    ["Retrieval", (r) => (r ? `${r.evidence.cases.length} cases` : "—"),
      (r) => (r?.latency_ms?.retrieval_ms != null ? `PASS · ${r.latency_ms.retrieval_ms} ms` : "PASS")],
    ["Evidence", (r) => (r ? `score ${num(r.evidence.quality)}` : "—"),
      (r) => (r?.latency_ms?.evidence_scoring_ms != null ? `PASS · ${r.latency_ms.evidence_scoring_ms} ms` : "PASS")],
    ["Generation", (r) => {
        if (!r) return "—";
        const st = r.response?.generation_status;
        if (st === "FAILED") return "FAILED";
        if (st === "SKIPPED") return "skipped";
        return r.response?.is_mock ? "mock draft" : "LLM draft";
      },
      (r) => {
        const ms = r?.latency_ms?.generation_ms;
        const st = r?.response?.generation_status;
        if (st === "FAILED") return ms != null ? `FAILED · ${ms} ms` : "FAILED";
        if (st === "SKIPPED") return "SKIPPED";
        return ms != null ? `PASS · ${ms} ms` : "PASS";
      }],
    ["Grounding", (r) => (r && r.response.grounding_score != null ? `score ${num(r.response.grounding_score)}` : "n/a"),
      (r) => (r && r.response.grounding_score != null ? "PASS" : "NOT RUN")],
    ["Decision", (r) => (r ? r.decision.decision : "—"),
      (r) => (r ? (r.decision.decision === "AUTO" ? "AUTO-HANDLE" : "ESCALATE") : "—")],
  ];

  function initialPipeline() {
    return STAGES.map(([name]) => stageEl(name, "idle", "waiting"));
  }

  function stageEl(name, status, metric, detail) {
    return el(
      "div",
      { class: "stage" + (status === "idle" ? "" : " " + status), "data-stage": name },
      el("span", { class: "s-status" }),
      el("div", { class: "s-name", text: name }),
      el("div", { class: "s-metric", text: metric }),
      detail ? el("div", { class: "s-detail small text-muted", text: detail }) : null
    );
  }

  function setPipeline(state, result) {
    const stages = document.querySelectorAll("#pipeline .stage");
    if (state === "running") {
      stages.forEach((s) => { s.className = "stage running"; s.querySelector(".s-metric").textContent = "…"; });
      return;
    }
    if (state === "failed") {
      stages.forEach((s) => { s.className = "stage failed"; s.querySelector(".s-metric").textContent = "stopped"; });
      return;
    }
    stages.forEach((s, i) => {
      const [, metricFn, detailFn] = STAGES[i];
      const detailEl = s.querySelector(".s-detail");
      // A failed generation stage renders as FAILED (red), the rest as done.
      const isFailed = result && STAGES[i][0] === "Generation"
        && result.response?.generation_status === "FAILED";
      s.className = "stage " + (isFailed ? "failed" : "done");
      s.querySelector(".s-metric").textContent = metricFn(result);
      if (detailEl) detailEl.textContent = detailFn(result) || "";
    });
  }

  /* ---------- panes ---------- */

  function composePane() {
    const options = EXAMPLES.map((m) => el("option", { value: m, text: m.length > 46 ? m.slice(0, 46) + "…" : m }));
    return card(
      "Customer message",
      el(
        "div",
        null,
        el("textarea", {
          id: "msg-input",
          placeholder: "e.g. Why hasn't my refund arrived yet?",
          "aria-label": "Customer message",
        }),
        el("div", { class: "mt-3" },
          el("select", { "data-examples": true, "aria-label": "Example messages" },
            el("option", { value: "", text: "Load an example…" }), ...options)),
        el("div", { class: "row mt-3" },
          el("button", { class: "btn primary grow", "data-analyze": true, text: "Analyze Message" })),
        el("p", { class: "hint mt-3" }, "⌘/Ctrl + Enter to analyze. Retrieval runs over historically resolved AmazonHelp conversations.")
      )
    );
  }

  function analysisPane() {
    return el(
      "div",
      { class: "pane", id: "analysis-out" },
      emptyBox("No analysis yet", "Submit a message to see intent and evidence.")
    );
  }

  function decisionPane() {
    return el(
      "div",
      { class: "pane", id: "decision-out" },
      emptyBox("No decision yet", "The AUTO / ESCALATE decision and its reasons appear here.")
    );
  }

  /* ---------- result cards ---------- */

  function provenanceCard(result) {
    const rid = result.request_id || "—";
    const ridEl = el("span", { class: "mono", text: rid, title: "Click to copy the request ID",
      style: "cursor: pointer", onclick: () => navigator.clipboard?.writeText(rid) });
    const prov = result.provenance || {};
    return card("Request provenance",
      el("div", { class: "kv-grid" },
        el("div", null,
          el("div", { class: "row between small mb-3" },
            el("span", { class: "text-muted", text: "Request ID (click to copy)" }), ridEl),
          UI.kv("Provider", prov.provider ?? result.response?.provider ?? "—"),
          UI.kv("Mode", prov.mode ?? "—"),
          UI.kv("Model", prov.model ?? result.response?.model ?? "—"),
          UI.kv("Timestamp", prov.timestamp ?? "—")),
        el("div", null,
          UI.kv("Intent", `${result.intent.name} (${num(result.intent.confidence)})`),
          UI.kv("Evidence score", num(result.evidence.quality)),
          UI.kv("Grounding", result.response?.grounding_score != null ? num(result.response.grounding_score) : "not evaluated"),
          UI.kv("Decision", result.decision.decision))));
  }

  function intentCard(result) {
    const body = el(
      "div",
      null,
      el("div", { class: "row between" },
        intentBadge(result.intent.name),
        el("strong", { text: num(result.intent.confidence) })
      ),
      el("div", { class: "mt-3" }, confidenceMeter(result.intent.confidence)),
      el("div", { class: "mt-4 section-title", text: "Top alternative intents" }),
      altIntents(result.intent.all_scores || {}, result.intent.name),
      ambiguityBlock(result.ambiguity)
    );
    return card("Intent", body);
  }

  function altIntents(allScores, primary) {
    const alts = Object.entries(allScores).filter(([k]) => k !== primary).slice(0, 3);
    if (!alts.length) return el("div", { class: "hint", text: "No distribution available." });
    return el(
      "div",
      { class: "hbar-list" },
      alts.map(([name, p]) =>
        el("div", { class: "hbar-row" },
          el("span", { class: "name", text: name }),
          el("div", { class: "bar" }, el("span", { style: `width:${(p * 100).toFixed(1)}%` })),
          el("span", { class: "val", text: num(p) })
        ))
    );
  }

  function ambiguityBlock(amb) {
    if (!amb) return null;
    const flags = [];
    if (amb.is_ambiguous) flags.push("ambiguous intent");
    if (amb.multi_intent_suspected) flags.push("multi-intent");
    if (amb.sparse_evidence) flags.push("sparse evidence");
    if (amb.retrieval_disagreement) flags.push("retrieval disagreement");
    if (amb.noise_suspected) flags.push("noisy message");
    if (!flags.length) return null;
    return el(
      "div",
      { class: "mt-4" },
      el("div", { class: "row wrap" },
        el("span", { class: "small text-muted", text: "Signals:" }),
        flags.map((f) => badge(f, "amber")))
    );
  }

  function evidenceCard(result) {
    const cases = result.evidence.cases || [];
    const feats = result.evidence.features || {};
    const body = el("div");
    body.appendChild(
      el("div", { class: "row between mb-3" },
        el("span", { class: "small text-muted",
          text: `${cases.length} historical case(s) · evidence score ${num(result.evidence.quality)} · agreement ${(num(result.evidence.intent_agreement_rate ?? topAgreement(cases)))}` }),
        el("span", { class: "small", style: "font-variant-numeric: tabular-nums", text: `top sim ${num(cases[0]?.similarity)}` })
      )
    );

    if (!cases.length) {
      body.appendChild(emptyBox("No historical evidence retrieved",
        "The system will not invent an answer — this routes to escalation."));
      return card("Retrieved historical evidence", body);
    }

    for (const c of cases) {
      body.appendChild(
        el(
          "div",
          { class: "evidence-card" },
          el("div", { class: "ev-head" },
            el("div", { class: "row" }, intentBadge(c.intent),
              el("span", { class: "small text-muted", text: c.conversation_id })),
            el("div", { class: "row" },
              el("span", { class: "badge blue", text: `sim ${num(c.similarity)}`,
                tip: "Cosine similarity between this historical message and the customer's message." }))
          ),
          el(
            "div",
            { class: "ev-body" },
            el("div", { class: "quote" }, el("span", { class: "who", text: "Customer (historical)" }), c.customer_message),
            c.resolution
              ? el("div", { class: "quote" }, el("span", { class: "who", text: "Resolution (historical)" }), c.resolution)
              : el("div", { class: "hint" }, "No recorded resolution for this case.")
          )
        )
      );
    }

    body.appendChild(
      el("div", { class: "mt-4 section-title", text: "Evidence score composition" }),
      featBars(feats)
    );
    return card("Retrieved historical evidence", body);
  }

  function topAgreement(cases) {
    if (!cases.length) return 0;
    const top = cases[0].intent;
    return cases.filter((c) => c.intent === top).length / cases.length;
  }

  function featBars(feats) {
    const rows = Object.entries(feats);
    if (!rows.length) return el("div", { class: "hint", text: "Feature breakdown unavailable." });
    return el(
      "div",
      { class: "hbar-list" },
      rows.map(([k, v]) =>
        el("div", { class: "hbar-row" },
          el("span", { class: "name", text: k }),
          el("div", { class: "bar" }, el("span", { style: `width:${(v * 100).toFixed(1)}%` })),
          el("span", { class: "val", text: num(v) })
        ))
    );
  }

  function generationCard(result) {
    const resp = result.response || {};
    const body = el("div");
    if (resp.is_mock) {
      body.appendChild(
        el("div", { class: "mock-warning mb-3" },
          el("span", { class: "icon", text: "⚠" }),
          el("span", { text:
            "MOCK RESPONSE — generated by the deterministic offline provider, not a real LLM. " +
            "No external API call was made. This draft demonstrates the pipeline shape only." })
        )
      );
    }
    // Polished generation-failure state: honest, inspectable, retryable.
    if (resp.generation_status === "FAILED") {
      body.appendChild(
        el("div", { class: "callout danger mb-3" },
          el("div", { class: "callout-title", text: "GENERATION FAILED" }),
          el("p", { class: "small text-secondary" },
            (result.response.provider || "The configured provider") +
            " could not produce a valid response. The system escalated instead of sending an unsupported answer."),
          el("div", { class: "row wrap small mt-3" },
            badge(`reason: ${resp.generation_error_code || "GENERATION_FAILED"}`, "red"),
            badge(`provider: ${result.response.provider || "—"}`, "gray"),
            badge(`status: unhealthy for this request`, "red"),
            result.request_id ? badge(`request: ${result.request_id.slice(0, 13)}…`, "gray") : null),
          resp.generation_error
            ? el("p", { class: "small text-muted mt-3", text: `Detail: ${resp.generation_error}` })
            : null));
      body.appendChild(emptyBox("No draft generated",
        "The escalation decision below explains why nothing was sent."));
      return card("Draft response", body);
    }
    if (!resp.draft) {
      body.appendChild(emptyBox("No draft generated",
        "Generation failed or was skipped; the decision explains why."));
      return card("Draft response", body);
    }
    body.appendChild(el("div", { class: "quote", style: "border-left: 3px solid var(--accent-border); padding-left: 12px", text: resp.draft }));
    return card("Draft response", body);
  }

  function decisionCard(result) {
    const d = result.decision || {};
    const isAuto = d.decision === "AUTO";
    const banner = el(
      "div",
      { class: "decision-banner " + (isAuto ? "auto" : "escalate") },
      el("div", { class: "d-label", text: "Decision" }),
      el("div", { class: "d-value", text: isAuto ? "AUTO-HANDLE" : "ESCALATE" }),
      el("div", { class: "row mt-3" },
        badge(`risk: ${d.risk_level ?? "—"}`, d.risk_level === "high" ? "red" : d.risk_level === "medium" ? "amber" : "green"),
        badge(`internal: ${d.internal_decision ?? "—"}`, "gray",
          "Internal three-state model (AUTO/REVIEW/ESCALATE) collapsed to the public two-state output.")
      ),
      el("p", { class: "small text-secondary mt-3", text: d.reason || "" })
    );

    const reasons = (d.reason_codes || []).map((code) =>
      el("li", { class: "bad" }, el("span", { class: "bullet", text: "!" }), el("span", null, el("span", { class: "mono-badge", text: code })))
    );
    const reasonsCard = reasons.length
      ? card("Why? — reason codes", el("ul", { class: "reason-list" }, reasons))
      : card("Why?", el("ul", { class: "reason-list" },
          el("li", { class: "good" }, el("span", { class: "bullet", text: "✓" }),
            el("span", { text: "All policy gates passed: strong evidence, confident intent, grounded claims, low-risk intent." }))));

    const amb = result.ambiguity;
    const ambCard = amb && (amb.is_ambiguous || amb.multi_intent_suspected)
      ? card("Ambiguity signals", el(
          "ul", { class: "reason-list" },
          amb.is_ambiguous ? el("li", { class: "bad" }, el("span", { class: "bullet", text: "!" }),
            el("span", { text: `Top-2 intent margin ${num(amb.top2_margin)} — genuinely ambiguous between intents.` })) : null,
          amb.multi_intent_suspected ? el("li", { class: "bad" }, el("span", { class: "bullet", text: "!" }),
            el("span", { text: "Multiple intents detected: " + amb.multi_intent_candidates.join(", ") + ". Multi-intent → escalate rather than guess." })) : null
        ))
      : null;

    const latency = result.latency_ms || {};
    const latencyRow = el("div", { class: "row wrap mt-3" },
      Object.entries(latency).map(([k, v]) => badge(`${k.replace("_ms", "")} ${v}ms`, "gray")));

    return el("div", { class: "pane" }, banner, el("div", { class: "mt-4" }, reasonsCard, ambCard, el("div", { class: "mt-3" }, latencyRow)));
  }

  function groundingCard(result) {
    const resp = result.response || {};
    const body = el("div");

    if (resp.grounding_score == null) {
      body.appendChild(el("div", { class: "hint", text:
        "Grounding not evaluated for this decision path (generation skipped or failed). In mock mode, " +
        "generation-quality metrics are NOT AVAILABLE IN MOCK MODE." }));
      return card("Grounding", body);
    }

    body.appendChild(el("div", { class: "row between mb-3" },
      el("strong", { text: num(resp.grounding_score) }),
      resp.grounded ? badge("verification passed", "green") : badge("verification failed", "red")
    ));
    body.appendChild(meter(resp.grounding_score, resp.grounded ? "green" : "red"));

    const claims = resp.grounded_claims || [];
    const unsupported = resp.unsupported_claims || [];
    if (claims.length || unsupported.length) {
      body.appendChild(el("div", { class: "mt-4 section-title", text: "Claims" }));
      body.appendChild(el("div", { class: "claim-list" },
        claims.map((c) => el("div", { class: "claim ok", text: "✓ " + c })),
        unsupported.map((c) => el("div", { class: "claim bad", text: "✗ " + c }))
      ));
    } else {
      body.appendChild(el("div", { class: "hint mt-3", text:
        "No specific claims returned to verify (or the model correctly abstained due to insufficient evidence)." }));
    }
    return card("Grounding", body);
  }

  window.Pages = window.Pages || {};
  window.Pages.agent = render;
})();
