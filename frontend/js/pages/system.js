/* System — "How does the system work?" Includes honest current limitations. */
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
          "Runtime status and architecture. The classifier, retrieval index, and evidence scoring are real trained ",
          "components; the LLM draft is mock unless a live provider is configured and shown as such.")),
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
    const reachable = providerHealth?.reachable;
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
      modeLabel = `LIVE (${sys?.llm_provider}) — UNHEALTHY`;
      modeBadge = badge("unhealthy", "red");
    } else if (configured) {
      modeLabel = `LIVE (${sys?.llm_provider}) — not verified`;
      modeBadge = badge("configured", "gray");
    } else {
      modeLabel = "NOT CONFIGURED";
      modeBadge = badge("not configured", "red");
    }

    return card("Runtime status",
      el("div", { class: "kv-grid" },
        el("div", null,
          el("div", { class: "section-title", text: "Process" }),
          kvRow("Brand", sys?.brand ?? "—"),
          kvRow("Environment", sys?.app_env ?? "—"),
          kvRow("LLM provider", sys?.llm_provider ?? "—"),
          el("div", { class: "row between small mb-3" },
            el("span", { class: "text-muted", text: "Mode" }), modeBadge),
          kvRow("Mode detail", modeLabel),
          kvRow("Model name", model),
          !mock && latency != null ? kvRow("Health check latency", latency + " ms") : null,
          !mock && errorCode ? kvRow("Error code", errorCode) : null,
          !mock && providerHealth?.error_message ? kvRow("Error", providerHealth.error_message) : null),
        el("div", null,
          el("div", { class: "section-title", text: "Model artifacts" }),
          ...Object.entries(sys?.artifacts_present || {}).map(([k, v]) => artifact(k, v)),
          el("p", { class: "hint mt-3" }, "Artifacts are produced by scripts/train_and_save_agent.py from the pinned requirements in docs/reproducibility.md."))));
  }

  function kvRow(k, v) {
    return el("div", { class: "row between small mb-3" },
      el("span", { class: "text-muted", text: k }),
      el("span", { class: "mono", text: String(v) }));
  }

  function architectureCard() {
    const nodes = [
      ["Customer message", ""],
      ["API (Flask, request ID + structured logs)", ""],
      ["Intent Classifier — TF-IDF + LogisticRegression", ""],
      ["Retriever — TF-IDF cosine over resolved cases (temporal index only)", ""],
      ["Evidence Scorer — empirically calibrated weights", ""],
      ["LLM Generator — evidence-only prompt, JSON output (MOCK by default)", "acc"],
      ["Grounding Validator — independent claim verification", ""],
      ["Risk / Escalation Policy — multi-signal, named reason codes", ""],
      ["AUTO-HANDLE → send draft", "acc"],
      ["ESCALATE → human queue", "human"],
    ];
    return card("Decision flow",
      el("div", { class: "arch-flow" },
        nodes.map(([label, cls], i) =>
          el("div", null,
            el("div", { class: "arch-node " + cls, text: label }),
            i < nodes.length - 1 ? el("div", { class: "arch-arrow", text: "↓" }) : null))));
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
      ["OOD detection", "general_other is an uncertainty bucket, not a novelty detector. Experimental ambiguity signals added but not yet fully validated."],
      ["Multi-intent", "Detected via keyword-signal disagreement → escalates. True multi-label resolution is out of scope and not claimed."],
      ["Live LLM verification", "Generation + judge run in mock mode in this environment. Live Groq/Gemini paths are implemented and unit-tested but NOT verified against the real APIs."],
      ["Docker", "Docker build and container smoke test executed in this environment — see FINAL_VERIFICATION.md for the exact result."],
      ["Resolution-consistency calibration", "Feature weight calibrated to ~0 against the classifier-correctness proxy; runtime computation now wired, but the right calibration target (groundedness) needs live LLM data."],
      ["Single labeler", "The golden set had one primary labeler, not a multi-annotator panel; inter-annotator agreement is unmeasured."],
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
