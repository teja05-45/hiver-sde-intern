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
    system: { title: "System", sub: "What is this built from?", render: () => Pages.system },
  };

  const root = document.getElementById("page-root");
  const pill = document.getElementById("mode-pill");
  const pillText = document.getElementById("mode-pill-text");

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

  /* ---------------- Sidebar collapse + keyboard navigation ---------------- */
  const KEY_ORDER = ["overview", "agent", "evaluation", "failures", "golden", "judge", "decisions", "system"];

  function setupShell() {
    const shell = document.getElementById("shell");
    const sidebar = document.getElementById("sidebar");
    const collapseBtn = document.getElementById("collapse-btn");
    const backdrop = document.getElementById("sidebar-backdrop");
    if (!shell || !collapseBtn) return;

    function setCollapsed(collapsed) {
      shell.classList.toggle("collapsed", collapsed);
      collapseBtn.setAttribute("aria-expanded", String(!collapsed));
      collapseBtn.textContent = collapsed ? "»" : "«";
      collapseBtn.setAttribute("data-tip", collapsed ? "Expand sidebar ( ] )" : "Collapse sidebar ( [ )");
      try { localStorage.setItem("omniroute.sidebar", collapsed ? "1" : "0"); } catch (_) { /* private mode */ }
    }
    collapseBtn.addEventListener("click", () => setCollapsed(!shell.classList.contains("collapsed")));
    try { if (localStorage.getItem("omniroute.sidebar") === "1") setCollapsed(true); } catch (_) { /* ignore */ }

    /* Keyboard: [ and ] collapse; g then 1..8 jumps between pages. */
    let pendingG = false;
    document.addEventListener("keydown", (e) => {
      const target = e.target;
      const typing = target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.tagName === "SELECT" || target.isContentEditable);
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
      if (pendingG) {
        pendingG = false;
        const idx = parseInt(e.key, 10);
        if (idx >= 1 && idx <= KEY_ORDER.length) { location.hash = "#" + KEY_ORDER[idx - 1]; return; }
      }
      if (e.key === "[") { setCollapsed(true); }
      else if (e.key === "]") { setCollapsed(false); }
      else if (e.key === "g") { pendingG = true; setTimeout(() => { pendingG = false; }, 1200); }
      else if (e.key === "Escape" && window.innerWidth <= 900) { shell.classList.remove("open"); }
    });

    /* Mobile: nav items close the drawer; backdrop click closes it. */
    if (sidebar && backdrop) {
      sidebar.addEventListener("click", (e) => {
        if (e.target.closest(".nav-item") && window.innerWidth <= 900) shell.classList.remove("open");
      });
      backdrop.addEventListener("click", () => shell.classList.remove("open"));
    }
  }

  function initMobileToggle() {
    /* Hamburger button (created here to keep index.html clean). */
    const actions = document.querySelector(".topbar .actions");
    if (!actions || document.getElementById("nav-toggle")) return;
    const btn = document.createElement("button");
    btn.id = "nav-toggle";
    btn.className = "btn small nav-toggle";
    btn.setAttribute("aria-label", "Toggle navigation");
    btn.textContent = "☰";
    btn.addEventListener("click", () => document.getElementById("shell").classList.toggle("open"));
    actions.prepend(btn);
  }

  const MODE = {
    MOCK:      { cls: "mock",      label: "MOCK MODE",            tip: "Deterministic local provider. No external LLM/API call is being made." },
    LIVE:      { cls: "live",      label: "LIVE",                 tip: "Live provider verified with a real API call." },
    ERROR:     { cls: "unhealthy", label: "ERROR",                tip: "Provider is configured but failing. Real API calls will fail." },
    NOT_CONFIGURED: { cls: "mock", label: "NOT CONFIGURED",       tip: "No API key configured for the selected provider." },
    UNVERIFIED:{ cls: "unverified",label: "UNVERIFIED",           tip: "Provider is configured but its health could not be measured." },
    OFFLINE:   { cls: "unhealthy", label: "BACKEND OFFLINE",      tip: "Backend unreachable — start it with: cd backend && python -m app.api.app" },
  };

  function setPill(state, provider, model, errorCode) {
    const m = MODE[state] || MODE.UNVERIFIED;
    const prov = provider ? String(provider).toUpperCase() + " — " : "";
    let tip = m.tip;
    if (model) tip += ` Model: ${model}.`;
    if (errorCode) tip += ` Error code: ${errorCode}.`;
    pill.className = "mode-pill " + m.cls;
    pillText.textContent = state === "LIVE" || state === "ERROR" || state === "UNVERIFIED" ? prov + m.label : m.label;
    pill.setAttribute("data-tip", tip);
  }

  async function initStatus() {
    try {
      const [health, sys, providerHealth] = await Promise.all([
        API.health(),
        API.system().catch(() => null),
        fetch("/api/v1/provider/health?verify=1").then((r) => r.ok ? r.json() : null).catch(() => null),
      ]);
      const provider = sys?.llm_provider || health.llm_provider || "—";
      const isMock = health.mock_mode;

      /* Full provider state machine — the pill can only say LIVE after a
       * measured provider check succeeded. States: LIVE | MOCK |
       * NOT_CONFIGURED | ERROR | UNVERIFIED (health check itself failed) |
       * OFFLINE (backend unreachable, handled below). */
      if (isMock) {
        setPill("MOCK", provider, "mock-deterministic-v1");
      } else if (providerHealth && providerHealth.healthy === true) {
        setPill("LIVE", provider, providerHealth.model);
      } else if (providerHealth && providerHealth.configured === false) {
        setPill("NOT_CONFIGURED", provider);
      } else if (providerHealth && providerHealth.healthy === false) {
        setPill("ERROR", provider, null, providerHealth.error_code);
      } else {
        setPill("UNVERIFIED", provider);
      }
      if (typeof window.OverviewStatus !== "undefined") {
        const ok = isMock || (providerHealth && providerHealth.healthy === true);
        const bad = !providerHealth || providerHealth.healthy === false;
        window.OverviewStatus.set(ok ? "ok" : bad ? "err" : "warn",
          ok ? "Operational" : bad ? "Provider degraded" : "Unverified provider",
          ok ? (isMock ? "Running in mock mode — deterministic, offline." : "Provider health verified with a real API call.")
             : "Provider state could not be verified.");
      }

      const modeLabel = isMock ? "MOCK" : (providerHealth?.healthy ? "LIVE" : providerHealth?.healthy === false ? "ERROR" : "UNVERIFIED");
      document.getElementById("foot-brand").textContent = health.brand || "—";
      document.getElementById("foot-provider").textContent = provider;
      document.getElementById("foot-mode").textContent = modeLabel;
      document.getElementById("foot-env").textContent = health.app_env || "—";
    } catch (_) {
      setPill("OFFLINE");
      if (typeof window.OverviewStatus !== "undefined") {
        window.OverviewStatus.set("err", "Backend offline", MODE.OFFLINE.tip);
      }
    }
  }

  navigate();
  setupShell();
  initMobileToggle();
  initStatus();
})();
