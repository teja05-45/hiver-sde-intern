/* App shell logic: hash routing + global mode indicator. */
(function () {
  "use strict";

  const PAGES = {
    inbox:        { title: "Inbox",             sub: "What customer problem am I solving?",        render: () => Pages.inbox },
    assistant:    { title: "AI Assistant",      sub: "Analyze any customer message",               render: () => Pages.assistant },
    escalations:  { title: "Escalated",         sub: "Conversations waiting for a human",          render: () => Pages.escalations },
    resolved:     { title: "Resolved",          sub: "Conversations the AI handled",               render: () => Pages.resolved },
    overview:     { title: "Overview",          sub: "Operational health and automation quality",  render: () => Pages.overview },
    aiperformance:{ title: "AI Performance",    sub: "Intent and automation quality, verified",    render: () => Pages.aiperformance },
    retrieval:    { title: "Retrieval Quality", sub: "Is the historical evidence the right one?",  render: () => Pages.retrieval },
    failures:     { title: "Failure Analysis",  sub: "How the AI fails, from real cases",          render: () => Pages.failures },
    golden:       { title: "Golden Set",        sub: "The human-verified benchmark",               render: () => Pages.golden },
    judge:        { title: "LLM Judge",         sub: "Can response quality be measured?",          render: () => Pages.judge },
    decisions:    { title: "Decision Log",      sub: "Every agent decision, traceable",            render: () => Pages.decisions },
    system:       { title: "System",            sub: "Live state of every component",              render: () => Pages.system },
  };

  const root = document.getElementById("page-root");
  const pill = document.getElementById("mode-pill");
  const pillText = document.getElementById("mode-pill-text");

  function currentRoute() {
    const hash = (location.hash || "#inbox").replace("#", "");
    return PAGES[hash] ? hash : "inbox";
  }

  function navigate() {
    const route = currentRoute();
    const page = PAGES[route];

    document.querySelectorAll(".nav-item").forEach((n) =>
      n.classList.toggle("active", n.dataset.page === route));

    document.getElementById("page-title").textContent = page.title;
    document.getElementById("page-sub").textContent = page.sub;
    document.title = `${page.title} — EvidenceDesk`;

    root.replaceChildren();
    page.render()(root);
    root.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }

  window.addEventListener("hashchange", navigate);

  /* ---------------- Sidebar collapse + keyboard navigation ---------------- */
  const KEY_ORDER = Object.keys(PAGES);

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
      try { localStorage.setItem("evidencedesk.sidebar", collapsed ? "1" : "0"); } catch (_) { /* private mode */ }
    }
    collapseBtn.addEventListener("click", () => setCollapsed(!shell.classList.contains("collapsed")));
    try { if (localStorage.getItem("evidencedesk.sidebar") === "1") setCollapsed(true); } catch (_) { /* ignore */ }

    /* Keyboard: [ and ] collapse; g then 1..9 jumps between pages. */
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
      document.getElementById("foot-brand").textContent = health.product?.name || "EvidenceDesk";
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
