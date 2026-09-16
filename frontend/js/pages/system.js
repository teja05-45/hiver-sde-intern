/* System — runtime health + interactive architecture with implementation
 * file references. Honest limitations stated inline. */
(function () {
  "use strict";
  const { el, errorBox, card, badge } = UI;

  async function render(root) {
    let sys = null;
    let providerHealth = null;
    try { sys = await API.system(); } catch (_) { /* non-fatal */ }
    try {
      providerHealth = await fetch("/api/v1/provider/health?verify=1").then((r) => r.ok ? r.json() : null);
    } catch (_) { /* non-fatal */ }

    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "System" }),
        el("p", { class: "desc" },
          "Runtime health and architecture. The classifier, retrieval index, and evidence scoring are real trained ",
          "components; provider state below is measured (the LIVE badge only appears after a real API check succeeds).")),
      runtimeCard(sys, providerHealth),
      architectureCard(),
      dataPipelineCard(),
      limitationsCard()
    );
  }

  function runtimeCard(sys, providerHealth) {
    const artifact = (name, present) => el("div", { class: "row between small" },
      el("span", { class: "mono", text: name }),
      present ? badge("present", "green") : badge("missing", "red"));

    const mock = sys?.mock_mode;
    const configured = providerHealth?.configured ?? sys?.provider_status?.configured ?? false;
    const healthy = providerHealth?.healthy;
    const latency = providerHealth?.latency_ms;
    const errorCode = providerHealth?.error_code;
    const model = providerHealth?.model ?? sys?.model_name ?? "—";

    let modeLabel, modeBadge;
    if (mock) {
      modeLabel = "MOCK (deterministic, offline)";
      modeBadge = badge("mock", "amber");
    } else if (healthy === true) {
      modeLabel = `LIVE (${sys?.llm_provider})`;
      modeBadge = badge("healthy", "green");
    } else if (healthy === false) {
      modeLabel = `LIVE (${sys?.llm_provider}) — ERROR`;
      modeBadge = badge("error", "red");
    } else if (configured) {
      modeLabel = `LIVE (${sys?.llm_provider}) — UNVERIFIED`;
      modeBadge = badge("unverified", "gray");
    } else {
      modeLabel = "NOT CONFIGURED";
      modeBadge = badge("not configured", "red");
    }

    const components = sys?.components || {};
    const compRow = (key, label) => {
      const c = components[key];
      if (!c) return null;
      const ok = ["healthy", "loaded", "available"].includes(String(c.status));
      const warn = /mock|configured|not configured/i.test(String(c.status));
      return el("div", { class: "row between small" },
        el("span", { class: "text-muted", text: label }),
        el("span", { class: "row" },
          warn ? badge(c.status, "amber") : badge(c.status, ok ? "green" : "red")));
    };

    return card("System health",
      el("div", { class: "kv-grid" },
        el("div", null,
          el("div", { class: "section-title", text: "Components" }),
          compRow("api", "Backend API"),
          compRow("classifier", "Classifier"),
          compRow("retriever", "Retriever"),
          compRow("evidence_model", "Evidence scorer"),
          compRow("llm_provider", "LLM provider"),
          compRow("golden_set", "Golden set"),
          !mock && latency != null ? el("div", { class: "row between small" },
            el("span", { class: "text-muted", text: "Provider check latency" }),
            el("span", { class: "mono", text: latency + " ms" })) : null,
          !mock && errorCode ? el("div", { class: "row between small" },
            el("span", { class: "text-muted", text: "Provider error" }),
            el("span", { class: "mono", text: errorCode })) : null),
        el("div", null,
          el("div", { class: "section-title", text: "Process" }),
          kvRow("Brand", sys?.brand ?? "—"),
          kvRow("Environment", sys?.app_env ?? "—"),
          kvRow("LLM provider", sys?.llm_provider ?? "—"),
          el("div", { class: "row between small mb-3" },
            el("span", { class: "text-muted", text: "Mode" }), modeBadge),
          kvRow("Mode detail", modeLabel),
          kvRow("Model", model),
          el("div", { class: "section-title", text: "Model artifacts" }),
          ...Object.entries(sys?.artifacts_present || {}).map(([k, v]) => artifact(k, v)),
          el("p", { class: "hint mt-3" }, "Artifacts are produced by scripts/train_and_save_agent.py from the pinned requirements in docs/reproducibility.md."))));
  }

  function kvRow(k, v) {
    return el("div", { class: "row between small mb-3" },
      el("span", { class: "text-muted", text: k }),
      el("span", { class: "mono", text: String(v) }));
  }

  /* Interactive architecture: each node is clickable and shows what it does
   * plus the actual implementation file. */
  function architectureCard() {
    const nodes = [
      ["Request", "POST /api/v1/agent/respond with the customer message; every request gets an X-Request-ID and structured PII-free logs.", "backend/app/api/app.py"],
      ["Intent Classifier", "TF-IDF (1–2 grams) + multinomial logistic regression over 11 effective intents; returns the full probability distribution, whose top-2 margin feeds ambiguity detection.", "backend/app/classification/baselines.py"],
      ["Retriever", "TF-IDF cosine over ~17k resolved conversations. Temporal discipline: the index only ever contains conversations earlier than the query period.", "backend/app/retrieval/retriever.py"],
      ["Evidence Scorer", "Combines top similarity, intent agreement, resolution consistency, and case count with empirically fitted weights (classifier confidence deliberately excluded).", "backend/app/services/evidence_scoring.py"],
      ["Novelty / OOD", "Derives an out-of-distribution score from measured signals: saturated confidence, top-2 margin, nearest-neighbor similarity, retrieval agreement. ESCALATE-only veto.", "backend/app/services/novelty.py"],
      ["LLM Generator", "Evidence-only prompt in strict JSON (system policy / message / evidence as separate fields — prompt-injection defense); must abstain when evidence is insufficient.", "backend/app/generation/generator.py"],
      ["Claim Verifier", "Extracts claims from the ACTUAL draft and verifies each against the retrieved evidence with matched case IDs; the LLM's self-report is never trusted.", "backend/app/generation/claims.py"],
      ["Escalation Policy", "Multi-signal decision with named reason codes: hard fail-safes first (privacy, OOD, multi-intent, unsupported claims, generation failure), then evidence/confidence/grounding thresholds.", "backend/app/escalation/policy.py"],
      ["AUTO → send", "All gates passed: strong evidence, confident intent, verified claims, low-risk intent.", "backend/app/escalation/policy.py"],
      ["ESCALATE → human", "Routed to a human with the reason codes attached — the audit trail a reviewer needs.", "backend/app/escalation/policy.py"],
    ];
    const detail = el("div", { class: "pipe-detail", role: "status" });
    const show = (i) => detail.replaceChildren(
      el("p", { class: "small text-secondary", text: nodes[i][1] }),
      el("p", { class: "mono small mt-2", style: "color: var(--accent-hover)", text: nodes[i][2] }));
    const flow = el("div", { class: "arch-flow" },
      nodes.map(([label], i) => {
        const cls = i === 8 ? "acc" : i === 9 ? "human" : "";
        const btn = el("button", {
          class: "arch-node " + cls + (i === 0 ? " active" : ""),
          text: label,
          onclick: (e) => {
            flow.querySelectorAll(".arch-node").forEach((n) => n.classList.remove("active"));
            e.currentTarget.classList.add("active");
            show(i);
          },
        });
        return i < nodes.length - 1
          ? el("div", null, btn, el("div", { class: "arch-arrow", text: "↓" }))
          : el("div", null, btn);
      }));
    show(0);
    return card("Decision flow — click any stage",
      el("div", { class: "grid cols-2" },
        flow,
        el("div", null, detail)));
  }

  function dataPipelineCard() {
    const stages = [
      ["scripts/profile_dataset.py", "brand profiling → AmazonHelp selected on data"],
      ["scripts/build_dataset.py", "60k sampled AmazonHelp conversations reconstructed"],
      ["scripts/discover_intents.py", "TF-IDF + KMeans → human-merged 12-intent taxonomy"],
      ["scripts/build_labeled_dataset.py", "silver labels + temporal train/dev/test split"],
      ["scripts/evaluate_baselines.py", "majority + TF-IDF/LogReg baselines"],
      ["scripts/evaluate_retrieval.py", "Recall@k / MRR with temporal index"],
      ["scripts/calibrate_evidence_score.py", "empirical evidence-score weights (no hand-picking)"],
      ["scripts/evaluate_automation.py", "precision/coverage curve on silver + golden check"],
      ["scripts/build_golden_set.py + apply_golden_corrections.py", "200 human-verified examples"],
      ["scripts/evaluate_against_golden.py", "the honest accuracy number"],
      ["scripts/analyze_failures.py", "failure taxonomy from real misclassifications"],
      ["scripts/train_and_save_agent.py", "production artifacts in models/"],
    ];
    return card("Data & evaluation pipeline",
      el("table", { class: "data" },
        el("thead", null, el("tr", null, el("th", { text: "Stage" }), el("th", { text: "Purpose" }))),
        el("tbody", null, stages.map(([s, p]) => el("tr", null,
          el("td", null, el("span", { class: "mono", text: s })),
          el("td", { class: "small text-secondary", text: p }))))));
  }

  function limitationsCard() {
    const items = [
      ["TF-IDF semantic ceiling", "No paraphrase matching; low Recall@1 (exact value on the Evaluation page) bounds evidence quality. Sentence embeddings are the known fix (interface ready)."],
      ["Classifier degraded by the remap", "The 2026-09-12 cluster→intent remap left specific intents with as few as 6 training examples; golden accuracy is 19.5%. Retraining from the pre-remap mapping is the real fix and is recorded, not silently applied."],
      ["Golden-set precision CI", "Auto-precision is measured on only 32 golden auto-handled cases (95% CI 51–82%) — too wide to pin a deployment threshold."],
      ["Grounding is literal", "Word-overlap verification, not semantic entailment: paraphrases can false-accept or false-reject. The claim verifier distinguishes assertions from suggestions to reduce false rejects."],
      ["Ambiguity/OOD signals are experimental", "Direction-tested in unit tests and against the golden probes; not yet validated against a labeled ground truth. They only ever escalate, never force AUTO."],
      ["Single labeler", "The golden set had one primary labeler; no inter-annotator agreement measurement."],
      ["Docker smoke-tested only", "Built and boot-verified in this environment; not load-tested or production-hardened."],
    ];
    return card("Current limitations (stated, not hidden)",
      el("ul", { class: "reason-list" },
        items.map(([t, d]) => el("li", null,
          el("span", { class: "bullet", text: "!" }),
          el("span", null, el("strong", { text: t + " — " }), el("span", { class: "text-secondary", text: d }))))));
  }

  window.Pages = window.Pages || {};
  window.Pages.system = render;
})();
