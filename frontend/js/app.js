/* App shell logic: hash routing + global mode indicator. */
(function () {
  "use strict";

  const PAGES = {
    overview: { title: "Overview", sub: "Can I trust this system?", render: () => Pages.overview },
    agent: { title: "Live Agent", sub: "Why did the system make this decision?", render: () => Pages.agent },
    evaluation: { title: "Evaluation", sub: "How well does it actually perform?", render: () => Pages.evaluation },
    failures: { title: "Failure Analysis", sub: "Where does it fail?", render: () => Pages.failures },
    golden: { title: "Golden Set", sub: "Are the evaluation labels trustworthy?", render: () => Pages.golden },
    judge: { title: "LLM Judge", sub: "Can I trust the evaluator?", render: () => Pages.judge },
    decisions: { title: "Decision Log", sub: "Why was the system designed this way?", render: () => Pages.decisions },
    system: { title: "System", sub: "How does the system work?", render: () => Pages.system },
  };

  const root = document.getElementById("page-root");

  function currentRoute() {
    const hash = (location.hash || "#overview").replace("#", "");
    return PAGES[hash] ? hash : "overview";
  }

  function navigate() {
    const route = currentRoute();
    const page = PAGES[route];

    document.querySelectorAll(".nav-item").forEach((n) =>
      n.classList.toggle("active", n.dataset.page === route));

    document.getElementById("page-title").textContent = page.title;
    document.getElementById("page-sub").textContent = page.sub;
    document.title = `${page.title} — Omniroute`;

    root.replaceChildren();
    page.render()(root);
    root.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }

  window.addEventListener("hashchange", navigate);

  async function initStatus() {
    const pill = document.getElementById("mode-pill");
    const pillText = document.getElementById("mode-pill-text");
    try {
      const [health, sys, providerHealth] = await Promise.all([
        API.health(),
        API.system().catch(() => null),
        fetch("/api/v1/provider/health?verify=1").then((r) => r.ok ? r.json() : null).catch(() => null),
      ]);
      const provider = sys?.llm_provider || health.llm_provider || "—";
      const isMock = health.mock_mode;

      // Determine the mode pill state:
      // - MOCK: deterministic local provider
      // - LIVE: live provider confirmed healthy
      // - UNHEALTHY: live provider configured but failing
      // - NOT CONFIGURED: no API key set
      let pillClass, pillLabel, pillTip;
      if (isMock) {
        pillClass = "mock";
        pillLabel = "MOCK MODE";
        pillTip = "Deterministic local provider. No external LLM/API call is being made.";
      } else if (providerHealth && providerHealth.healthy === false) {
        pillClass = "unhealthy";
        pillLabel = `${String(provider).toUpperCase()} — UNHEALTHY`;
        pillTip = `Provider ${provider} is configured but not responding correctly. Error: ${providerHealth.error_code || "unknown"}. Real API calls may fail.`;
      } else if (providerHealth && providerHealth.healthy === true) {
        pillClass = "live";
        pillLabel = `${String(provider).toUpperCase()} — LIVE`;
        pillTip = `Live provider: ${provider}. Real API calls are being made. Model: ${providerHealth.model || "default"}.`;
      } else if (providerHealth && providerHealth.configured === false) {
        pillClass = "mock";
        pillLabel = "NOT CONFIGURED";
        pillTip = "No API key configured for the selected provider. Using mock mode.";
      } else {
        // Health check not available — fall back to configured-but-unverified
        pillClass = "live";
        pillLabel = `LIVE — ${String(provider).toUpperCase()}`;
        pillTip = `Provider ${provider} is configured. Run provider health check to verify.`;
      }

      pill.className = "mode-pill " + pillClass;
      pillText.textContent = pillLabel;
      pill.setAttribute("data-tip", pillTip);

      document.getElementById("foot-brand").textContent = health.brand || "—";
      document.getElementById("foot-provider").textContent = provider;
      document.getElementById("foot-mode").textContent = isMock ? "MOCK" : (providerHealth?.healthy ? "LIVE" : providerHealth?.healthy === false ? "UNHEALTHY" : "LIVE");
      document.getElementById("foot-env").textContent = health.app_env || "—";
    } catch (_) {
      pill.className = "mode-pill mock";
      pillText.textContent = "BACKEND OFFLINE";
      pill.setAttribute("data-tip", "Backend unreachable — start it with: cd backend && python -m app.api.app");
    }
  }

  navigate();
  initStatus();
})();
