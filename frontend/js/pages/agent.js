/* Live Agent — "Why did the system make this decision?"
 * 40/60 agent workspace: composer left, decision + evidence trail right.
 * A horizontal pipeline timeline marks stages as the single /respond call
 * resolves; every stage's final label shows the REAL latency reported by
 * the API (never faked timing). Claim verification renders per-claim
 * objects derived from the actual draft text (backend/app/generation/claims.py).
 */
(function () {
  "use strict";
  const { el, num, badge, intentBadge, meter, errorBox, emptyBox, card } = UI;

  const EXAMPLES = [
    "My package says delivered but I never received it.",
    "Why hasn't my refund arrived yet?",
    "I was charged twice for my order.",
    "My account was hacked and I cannot log in.",
    "The app crashes every time I open my order history.",
    "My package arrived late, the product is damaged, and I want a refund.",
    "What is the capital of India?",
    "I need help.",
  ];

  /* Human-readable narrative per reason code — one place, kept in sync with
   * backend/app/escalation/policy.py. Unknown codes fall back to the raw code. */
  const REASON_STORIES = {
    UNSUPPORTED_CLAIMS: "The draft contains a statement that no historical case supports. Sending it would mean asserting something we cannot back with evidence, so the system abstains and hands off to a human.",
    GENERATION_FAILED: "The language model failed or returned an invalid response. The system never guesses or silently substitutes a canned answer — it escalates.",
    RETRIEVAL_FAILED: "Historical evidence could not be retrieved, so there is no way to verify any draft. Escalation is the only safe path.",
    MULTI_INTENT: "The message raises several distinct issues at once. Answering only one of them would look like progress while the others go unaddressed, so a human reads the whole message.",
    OOD_REQUEST: "The message does not resemble anything in the support taxonomy — forcing it into a guessed intent would produce a confident, wrong answer.",
    PRIVACY_RISK: "The message appears to contain credentials (password, code, card number). A human must handle account security, and no automated draft should ever echo them.",
    HIGH_RISK_INTENT: "This intent is financially or security-sensitive (billing, account access, refunds, complaints). Policy escalates these regardless of how strong the evidence looks.",
    LOW_INTENT_CONFIDENCE: "The classifier is not confident enough in its intent prediction to auto-select a response template.",
    LOW_EVIDENCE_SCORE: "The retrieved historical evidence is too weak to ground a response.",
    INSUFFICIENT_EVIDENCE_COUNT: "Too few historical cases were retrieved to establish a consistent resolution pattern.",
    RETRIEVAL_DISAGREEMENT: "Historical evidence mostly points to a different intent than the classifier chose. The disagreement itself is information — a human should see both signals.",
    WEAK_EVIDENCE: "Sparse historical corroboration for this message.",
    AMBIGUOUS_INTENT: "Two intents fit nearly equally well (top-2 margin below the project's own ambiguity threshold).",
    LOW_GROUNDING_SCORE: "Claim verification scored below the auto-handling threshold.",
    LOW_CLASSIFICATION_CONFIDENCE: "Classifier confidence below the auto-handling threshold.",
    NO_RESOLUTION_PATH: "No historical resolution pattern exists for this request.",
  };

  function render(root) {
    const compose = composePane();
    const analysis = analysisPane();
    const decision = decisionPane();

    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Live Agent" }),
        el("p", { class: "desc" },
          "Analyze a customer message and see why the system would answer or escalate. ",
          "Every claim in the draft is checked against real historical evidence before anything is sent.")),
      el("div", { class: "pipeline", id: "pipeline" }, initialPipeline()),
      el("div", { class: "agent-grid mt-4" },
        el("div", { class: "agent-compose" }, compose),
        el("div", { class: "agent-detail" }, analysis, decision))
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
        textarea.setAttribute("aria-invalid", "true");
        textarea.focus();
        return;
      }
      textarea.removeAttribute("aria-invalid");
      setPipeline("running");
      const out = document.getElementById("analysis-out");
      const dec = document.getElementById("decision-out");
      out.replaceChildren(el("div", { class: "skeleton block" }), el("div", { class: "skeleton block" }));
      dec.replaceChildren(el("div", { class: "skeleton block" }));
      const btn = compose.querySelector("[data-analyze]");
      btn.disabled = true;
      try {
        const result = await API.respond(message.trim(), 5);
        setPipeline("done", result);
        dec.replaceChildren(decisionCard(result), whyCard(result), groundingCard(result));
        out.replaceChildren(intentCard(result), evidenceCard(result), generationCard(result));
      } catch (err) {
        out.replaceChildren(errorBox(err));
        dec.replaceChildren(el("div"));
        setPipeline("failed");
      } finally {
        btn.disabled = false;
      }
    }
  }

  /* ---------- pipeline timeline ---------- */

  const STAGES = [
    ["Intent", (r) => r ? `${r.intent.name} · ${num(r.intent.confidence)}` : "—",
      (r) => lat("classification_ms", r)],
    ["Retrieval", (r) => r ? `${r.evidence.cases.length} cases` : "—",
      (r) => lat("retrieval_ms", r)],
    ["Evidence", (r) => r ? `score ${num(r.evidence.quality)}` : "—",
      (r) => lat("evidence_scoring_ms", r)],
    ["Generation", (r) => {
        if (!r) return "—";
        const st = r.response?.generation_status;
        if (st === "FAILED") return "FAILED";
        if (st === "SKIPPED") return "skipped";
        return r.response?.is_mock ? "mock draft" : "LLM draft";
      },
      (r) => lat("generation_ms", r, r?.response?.generation_status)],
    ["Grounding", (r) => {
        const cv = r?.response?.claim_verification;
        if (!r) return "—";
        if (cv == null) return "n/a";
        return `${cv.n_supported ?? 0}/${(cv.n_supported ?? 0) + (cv.n_unsupported ?? 0)} claims`;
      },
      (r) => (r?.response?.claim_verification ? "verified" : "not run")],
    ["Decision", (r) => (r ? r.decision.decision : "—"),
      (r) => (r ? (r.decision.decision === "AUTO" ? "auto-handle" : "escalate") : "—")],
  ];

  function lat(key, r, status) {
    const ms = r?.latency_ms?.[key];
    if (status === "FAILED") return ms != null ? `FAILED · ${ms} ms` : "FAILED";
    if (status === "SKIPPED") return "SKIPPED";
    return ms != null ? `${ms} ms` : "—";
  }

  function stageEl(name, status, metric, detail, index) {
    return el("div", { class: "stage" + (status === "idle" ? "" : " " + status), "data-stage": name },
      el("span", { class: "s-num", "aria-hidden": "true", text: String(index + 1).padStart(2, "0") }),
      el("span", { class: "s-status" }),
      el("div", { class: "s-name", text: name }),
      el("div", { class: "s-metric", text: metric }),
      detail ? el("div", { class: "s-detail small text-muted", text: detail }) : null);
  }

  function initialPipeline() {
    return STAGES.map(([name], i) => stageEl(name, "idle", "waiting", null, i));
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
      el("div", null,
        el("textarea", {
          id: "msg-input",
          placeholder: "e.g. My package says delivered but I never received it.",
          "aria-label": "Customer message",
        }),
        el("div", { class: "mt-3" },
          el("select", { "data-examples": true, "aria-label": "Example messages" },
            el("option", { value: "", text: "Load an example…" }), ...options)),
        el("div", { class: "row mt-3" },
          el("button", { class: "btn primary grow", "data-analyze": true, text: "Analyze message" })),
        el("p", { class: "hint mt-3" }, "Ctrl + Enter to analyze. Retrieval runs over historically resolved AmazonHelp conversations.")));
  }

  function analysisPane() {
    return el("div", { class: "pane", id: "analysis-out" },
      emptyBox("No analysis yet", "Submit a message to see intent, evidence, and the draft."));
  }

  function decisionPane() {
    return el("div", { class: "pane", id: "decision-out" },
      emptyBox("No decision yet", "The AUTO / ESCALATE decision and its reasoning appear here."));
  }

  /* ---------- result cards ---------- */

  function decisionCard(result) {
    const d = result.decision || {};
    const isAuto = d.decision === "AUTO";
    const codes = d.reason_codes || [];
    return el("div", { class: "decision-banner " + (isAuto ? "auto" : "escalate"), role: "status" },
      el("div", { class: "d-label", text: "Decision" }),
      el("div", { class: "d-value", text: isAuto ? "AUTO-HANDLE" : "ESCALATE" }),
      el("div", { class: "row wrap mt-2" },
        badge(`risk: ${d.risk_level ?? "—"}`,
          d.risk_level === "high" ? "red" : d.risk_level === "medium" ? "amber" : "green"),
        codes.length ? codes.map((c) => badge(c, "gray")) : badge("all gates passed", "green")),
      el("p", { class: "small text-secondary mt-3", text: d.reason || "" }));
  }

  /* The hero moment: a human-readable explanation of WHY. */
  function whyCard(result) {
    const d = result.decision || {};
    const codes = d.reason_codes || [];
    const primary = codes.length ? REASON_STORIES[codes[0]] : null;
    const parts = [];
    if (primary) {
      parts.push(el("p", { class: "why-main" }, primary));
    }
    const others = codes.slice(1).map((c) => REASON_STORIES[c]).filter(Boolean);
    if (others.length) {
      parts.push(el("ul", { class: "reason-list mt-3" },
        others.map((s) => el("li", { class: "bad" }, el("span", { class: "bullet", text: "!" }), el("span", { text: s })))));
    }
    if (!codes.length) {
      parts.push(el("p", { class: "why-main" },
        "Strong historical evidence, a confident intent match, verified claims, and a low-risk intent category — all gates passed, so the system answered automatically."));
    }
    const amb = result.ambiguity;
    const nov = result.novelty;
    if (nov?.is_ood) {
      parts.push(el("p", { class: "small text-muted mt-3" },
        `Novelty score ${num(nov.ood_score)} — nearest historical case similarity ${num(nov.subsignals?.top_similarity)}, retrieval agreement ${num(nov.subsignals?.retrieval_agreement)}.`));
    }
    if (amb?.retrieval_disagreement) {
      parts.push(el("p", { class: "small text-muted mt-3" },
        "Note: the retrieved historical cases mostly belong to a different intent than the classifier's prediction — both signals are shown above for the reviewing human."));
    }
    return card("Why?", el("div", null, parts));
  }

  function intentCard(result) {
    const alts = Object.entries(result.intent.all_scores || {})
      .filter(([k]) => k !== result.intent.name).slice(0, 3);
    const margin = result.ambiguity?.top2_margin;
    const body = el("div", null,
      el("div", { class: "row between" }, intentBadge(result.intent.name),
        el("strong", { text: num(result.intent.confidence) })),
      el("div", { class: "mt-3" }, confidenceMeter(result.intent.confidence)),
      margin != null ? el("div", { class: "small text-muted mt-2" }, `Top-2 margin: ${num(margin)}`) : null,
      alts.length ? el("div", { class: "mt-4 section-title", text: "Top alternatives" }) : null,
      alts.length ? el("div", { class: "hbar-list" },
        alts.map(([name, p]) => el("div", { class: "hbar-row" },
          el("span", { class: "name", text: name }),
          el("div", { class: "bar" }, el("span", { style: `width:${(p * 100).toFixed(1)}%` })),
          el("span", { class: "val", text: num(p) })))) : null,
      signalChips(result));
    return card("Intent", body);
  }

  function signalChips(result) {
    const amb = result.ambiguity;
    const nov = result.novelty;
    const chips = [];
    if (nov?.is_ood) chips.push(badge(`OOD ${num(nov.ood_score)}`, "red", "Message does not resemble the support taxonomy"));
    if (amb?.is_ambiguous) chips.push(badge("ambiguous", "amber", "Top-2 intent margin below the ambiguity threshold"));
    if (amb?.multi_intent_suspected) chips.push(badge("multi-intent", "red", "Multiple intents' signals matched this message"));
    if (amb?.retrieval_disagreement) chips.push(badge("retrieval disagreement", "amber", "Historical cases mostly disagree with the predicted intent"));
    if (amb?.noise_suspected) chips.push(badge("noisy", "gray", "Very short message; little signal to classify"));
    return chips.length ? el("div", { class: "row wrap mt-3" }, chips) : null;
  }

  function evidenceCard(result) {
    const cases = result.evidence.cases || [];
    const feats = result.evidence.features || {};
    const body = el("div");
    body.appendChild(el("div", { class: "row between mb-3" },
      el("span", { class: "small text-muted",
        text: `${cases.length} historical case(s) · evidence score ${num(result.evidence.quality)} · intent agreement ${num(result.evidence.intent_agreement_rate ?? 0)}` }),
      el("span", { class: "small", style: "font-variant-numeric: tabular-nums", text: `top sim ${num(cases[0]?.similarity)}` })));

    if (!cases.length) {
      body.appendChild(emptyBox("No historical evidence retrieved",
        "The system will not invent an answer — this routes to escalation."));
      return card("Retrieved historical evidence", body);
    }

    for (const c of cases) {
      body.appendChild(el("div", { class: "evidence-card" },
        el("div", { class: "ev-head" },
          el("div", { class: "row" }, intentBadge(c.intent),
            el("span", { class: "mono small text-muted", text: c.conversation_id })),
          el("span", { class: "badge blue", text: `sim ${num(c.similarity)}`,
            tip: "Cosine similarity between this historical message and the customer's message." })),
        el("div", { class: "ev-body" },
          el("div", { class: "quote" }, el("span", { class: "who", text: "Customer (historical)" }), c.customer_message),
          c.resolution
            ? el("div", { class: "quote" }, el("span", { class: "who", text: "Resolution (historical)" }), c.resolution)
            : el("div", { class: "hint" }, "No recorded resolution for this case."))));
    }

    body.appendChild(el("div", { class: "mt-4 section-title", text: "Evidence score composition" }), featBars(feats));
    return card("Retrieved historical evidence", body);
  }

  function featBars(feats) {
    const rows = Object.entries(feats);
    if (!rows.length) return el("div", { class: "hint", text: "Feature breakdown unavailable." });
    return el("div", { class: "hbar-list" },
      rows.map(([k, v]) => el("div", { class: "hbar-row" },
        el("span", { class: "name", text: k }),
        el("div", { class: "bar" }, el("span", { style: `width:${(v * 100).toFixed(1)}%` })),
        el("span", { class: "val", text: num(v) }))));
  }

  function generationCard(result) {
    const resp = result.response || {};
    const body = el("div");
    if (resp.is_mock) {
      body.appendChild(el("div", { class: "mock-warning mb-3" },
        el("span", { class: "icon", text: "⚠" }),
        el("span", { text:
          "MOCK RESPONSE — generated by the deterministic offline provider, not a real LLM. " +
          "No external API call was made. This draft demonstrates the pipeline shape only." })));
    }
    if (resp.generation_status === "FAILED") {
      body.appendChild(el("div", { class: "callout danger mb-3" },
        el("div", { class: "callout-title", text: "GENERATION FAILED" }),
        el("p", { class: "small text-secondary" },
          (result.response.provider || "The configured provider") +
          " could not produce a valid response. The system escalated instead of sending an unsupported answer."),
        el("div", { class: "row wrap small mt-3" },
          badge(`reason: ${resp.generation_error_code || "GENERATION_FAILED"}`, "red"),
          badge(`provider: ${result.response.provider || "—"}`, "gray"),
          result.request_id ? badge(`request: ${String(result.request_id).slice(0, 13)}…`, "gray") : null),
        resp.generation_error
          ? el("p", { class: "small text-muted mt-3", text: `Detail: ${resp.generation_error}` })
          : null));
      body.appendChild(emptyBox("No draft generated",
        "The escalation decision explains why nothing was sent."));
      return card("Draft response", body);
    }
    if (!resp.draft) {
      body.appendChild(emptyBox("No draft generated",
        "Generation failed or was skipped; the decision explains why."));
      return card("Draft response", body);
    }
    const prov = result.provenance || {};
    if (prov.provider && !resp.is_mock) {
      body.appendChild(el("p", { class: "hint mb-3" }, `Drafted by ${prov.provider} (${prov.model || "default model"}) — every claim below is verified against the historical evidence independently of the model.`));
    }
    body.appendChild(el("div", { class: "draft-quote", text: resp.draft }));
    return card("Draft response", body);
  }

  /* Per-claim verification — claims are substrings of the actual draft. */
  function groundingCard(result) {
    const resp = result.response || {};
    const cv = resp.claim_verification;
    const body = el("div");

    if (!cv) {
      body.appendChild(el("div", { class: "hint", text:
        "Claim verification did not run for this decision path (generation skipped or failed)." }));
      return card("Claim verification", body);
    }

    const total = (cv.n_supported ?? 0) + (cv.n_unsupported ?? 0);
    body.appendChild(el("div", { class: "row between mb-3" },
      el("strong", { text: total ? `${cv.n_supported}/${total} claims supported` : "No factual claims to verify" }),
      cv.passed ? badge("verification passed", "green") : badge("verification failed", "red")));
    if (total) body.appendChild(meter(cv.score, cv.passed ? "green" : "red"));

    const claims = cv.claims || [];
    if (!claims.length) {
      body.appendChild(el("div", { class: "hint mt-3", text: "The draft contained no sentences to verify." }));
      return card("Claim verification", body);
    }

    const statusMeta = {
      supported: { mark: "✓", cls: "ok", label: "supported" },
      unsupported: { mark: "✗", cls: "bad", label: "no evidence" },
      greeting: { mark: "·", cls: "neutral", label: "filler" },
      abstention: { mark: "⏸", cls: "abstain", label: "abstained" },
    };
    body.appendChild(el("div", { class: "claim-list mt-3" },
      claims.map((c) => {
        const meta = statusMeta[c.status] || statusMeta.unsupported;
        return el("div", { class: `claim ${meta.cls}` },
          el("div", { class: "row" },
            el("span", { class: "claim-mark", text: meta.mark, "aria-hidden": "true" }),
            el("span", { class: "claim-text", text: c.claim_text })),
          el("div", { class: "claim-meta small" },
            el("span", { class: "text-muted", text: meta.label }),
            c.evidence_ids && c.evidence_ids.length
              ? el("span", { class: "row wrap" }, c.evidence_ids.map((id) => el("span", { class: "mono-badge", text: id })))
              : null,
            el("span", { class: "text-muted", text: c.explanation })));
      })));
    return card("Claim verification — against the actual draft", body);
  }

  function confidenceMeter(value) {
    const tone = value >= 0.75 ? "green" : value >= 0.5 ? "amber" : "red";
    return meter(value, tone);
  }

  window.Pages = window.Pages || {};
  window.Pages.agent = render;
})();
