/* Inbox — the primary support workspace.
 * Three panes: ticket list (real dataset conversations), the customer
 * conversation (real thread), and the AI Copilot (intent, evidence,
 * suggested reply, decision). Every piece of AI output comes from
 * POST /api/v1/agent/respond; the ticket and conversation data come from
 * /api/v1/inbox and /api/v1/conversations/<id>. Nothing is invented here:
 * historical evidence cards quote real retrieved cases, and mock drafts
 * are visibly labeled by provenance.
 */
(function () {
  "use strict";
  const { el, num, badge, intentBadge, errorBox, emptyBox, card } = UI;

  /* Human-readable narrative per reason code — synced with
   * backend/app/escalation/policy.py; unknown codes fall back to raw. */
  const REASON_STORIES = {
    UNSUPPORTED_CLAIMS: "The draft contains a statement no historical case supports — sending it would assert something we cannot back with evidence.",
    GENERATION_FAILED: "The language model failed or returned invalid output. The system never guesses — it escalates.",
    RETRIEVAL_FAILED: "Historical evidence could not be retrieved, so no draft can be verified.",
    MULTI_INTENT: "The message raises several distinct issues at once; a human should read the whole message.",
    OOD_REQUEST: "The message does not resemble anything in the support taxonomy — forcing it into a guessed intent would produce a confident, wrong answer.",
    PRIVACY_RISK: "The message appears to contain credentials (password, code, card number) that must never be echoed by an automated reply.",
    HIGH_RISK_INTENT: "This intent is financially or security-sensitive; policy escalates these regardless of evidence strength.",
    LOW_INTENT_CONFIDENCE: "The classifier is not confident enough in the intent to answer automatically.",
    LOW_EVIDENCE_SCORE: "The retrieved historical evidence is too weak to ground a response.",
    INSUFFICIENT_EVIDENCE_COUNT: "Too few historical cases were found to establish a consistent resolution pattern.",
    RETRIEVAL_DISAGREEMENT: "Historical evidence mostly points to a different intent than the classifier chose.",
    WEAK_EVIDENCE: "Sparse historical corroboration for this message.",
    AMBIGUOUS_INTENT: "Two intents fit nearly equally well — the classifier cannot choose.",
    LOW_GROUNDING_SCORE: "Claim verification scored below the auto-handling threshold.",
    LOW_CLASSIFICATION_CONFIDENCE: "Classifier confidence below the auto-handling threshold.",
    NO_RESOLUTION_PATH: "No historical resolution pattern exists for this request.",
  };

  const state = { tickets: [], activeId: null, conversation: null, analysis: null, analyzing: false };

  function render(root) {
    root.replaceChildren(
      el("div", { class: "page-head" },
        el("h1", { text: "Inbox" }),
        el("p", { class: "desc" },
          "Real customer conversations from the AmazonHelp support corpus. Open a ticket, run the AI Copilot, ",
          "and see exactly which historical cases support the suggested reply — or why the AI escalates instead.")),
      el("div", { class: "inbox" },
        listPane(),
        conversationPane(),
        copilotPane())
    );

    loadTickets();
  }

  /* ================= LEFT: ticket list ================= */

  function listPane() {
    const search = el("input", {
      type: "text", placeholder: "Search messages…", "aria-label": "Search messages", id: "inbox-search",
    });
    const intentSel = el("select", { "aria-label": "Filter by intent" },
      el("option", { value: "", text: "All intents" }));
    const listBody = el("div", { class: "list-scroll", id: "ticket-list" },
      el("div", { class: "state-box", text: "Loading conversations…" }));

    // Debounced search (300ms) — avoids an API call per keystroke.
    let debounce = null;
    search.addEventListener("input", () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => loadTickets(), 300);
    });
    intentSel.addEventListener("change", () => loadTickets());

    API.intents().then((data) => {
      for (const i of data.intents || []) {
        intentSel.appendChild(el("option", { value: i.name, text: i.name }));
      }
    }).catch(() => { /* filter stays empty; list still works */ });

    return el("div", { class: "inbox-list" },
      el("div", { class: "card-body tight", style: "border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px;" },
        search,
        intentSel),
      listBody,
      el("div", { class: "card-body tight small text-muted", id: "inbox-count", style: "border-top: 1px solid var(--border);" }, "…"));
  }

  async function loadTickets() {
    const listBody = document.getElementById("ticket-list");
    const countEl = document.getElementById("inbox-count");
    if (!listBody) return;
    const q = document.getElementById("inbox-search")?.value || "";
    const intent = document.querySelector(".inbox-list select")?.value || "";
    listBody.replaceChildren(el("div", { class: "state-box", text: "Loading conversations…" }));
    try {
      const data = await API.inbox({ q, intent, page: 1, page_size: 25 });
      state.tickets = data.tickets || [];
      if (countEl) countEl.textContent = `${data.total} conversations · page 1 of ${data.pages}`;
      if (!state.tickets.length) {
        listBody.replaceChildren(emptyBox("No conversations match", "Try a different search or filter."));
        return;
      }
      listBody.replaceChildren(...state.tickets.map(ticketRow));
      // Auto-select the first ticket for a ready-to-explore workspace.
      if (!state.activeId) selectTicket(state.tickets[0].conversation_id, state.tickets[0]);
    } catch (err) {
      listBody.replaceChildren(errorBox(err));
    }
  }

  function ticketRow(t) {
    const row = el("button", { class: "ticket" + (t.priority === "high" ? " unread" : ""), "data-id": t.conversation_id, type: "button" },
      el("div", { class: "t-main" },
        el("div", { class: "t-top" },
          el("span", { class: "t-ref", text: t.customer_ref }),
          el("span", { class: "t-time", text: (t.created_at || "").slice(0, 10) })),
        el("div", { class: "t-preview", text: t.preview || "(no message text)" }),
        el("div", { class: "t-meta" },
          intentBadge(t.intent),
          t.priority === "high" ? badge("elevated", "amber", "Intent historically escalates — likely needs a human") : null,
          badge(t.status, t.status === "resolved" ? "green" : "blue"))));
    row.addEventListener("click", () => selectTicket(t.conversation_id, t));
    return row;
  }

  /* ================= MIDDLE: conversation ================= */

  function conversationPane() {
    return el("div", { class: "convo-pane" },
      el("div", { class: "convo-head", id: "convo-head" },
        el("div", { class: "hint", text: "Select a conversation from the inbox." })),
      el("div", { class: "convo-scroll", id: "convo-scroll" }),
      el("div", { class: "convo-composer", id: "convo-composer" }));
  }

  async function selectTicket(conversationId, ticket) {
    state.activeId = conversationId;
    document.querySelectorAll(".ticket").forEach((n) =>
      n.classList.toggle("active", n.dataset.id === conversationId));

    const head = document.getElementById("convo-head");
    const scroll = document.getElementById("convo-scroll");
    const composer = document.getElementById("convo-composer");
    head.replaceChildren(el("div", { class: "hint", text: "Loading conversation…" }));
    scroll.replaceChildren();
    composer.replaceChildren();

    try {
      const conv = await API.conversation(conversationId);
      state.conversation = conv;
      state.analysis = null;
      renderConversation(conv, ticket);
      renderCopilot(); // reset copilot for the new ticket
    } catch (err) {
      head.replaceChildren(errorBox(err));
    }
  }

  function renderConversation(conv, ticket) {
    const head = document.getElementById("convo-head");
    const scroll = document.getElementById("convo-scroll");
    const composer = document.getElementById("convo-composer");

    head.replaceChildren(
      el("div", { class: "row between wrap" },
        el("div", null,
          el("strong", { text: conv.customer_ref }),
          el("span", { class: "small text-muted", text: "  ·  " + conv.conversation_id })),
        el("div", { class: "row" },
          intentBadge(conv.intent),
          badge(conv.status, conv.status === "resolved" ? "green" : "blue"),
          badge(conv.priority === "high" ? "elevated priority" : "normal priority",
            conv.priority === "high" ? "amber" : "gray"))),
      el("div", { class: "small text-muted", text:
        (conv.created_at ? "Started " + conv.created_at.slice(0, 10) : "Historical conversation") +
        (conv.resolution ? " · resolved by an agent" : " · no recorded resolution") }));

    const msgs = conv.messages || [];
    scroll.replaceChildren(...msgs.map((m) =>
      el("div", { class: "msg " + (m.role === "agent" ? "agent" : "customer") },
        el("span", { class: "who", text: m.role === "agent" ? "Support agent" : conv.customer_ref }),
        el("div", { class: "bubble", text: m.text }))));

    // Composer: "AI suggested reply" + manual escalate.
    const ta = el("textarea", { "aria-label": "Reply draft", placeholder: "Write a reply, or ask the AI Copilot for a suggestion…" });
    const aiBtn = el("button", { class: "btn primary", type: "button", text: state.analyzing ? "Analyzing…" : "AI Assist" });
    const escalateBtn = el("button", { class: "btn", type: "button", text: "Escalate" });
    aiBtn.addEventListener("click", () => runCopilot());
    escalateBtn.addEventListener("click", () => {
      ta.value = "Escalated to a human reviewer — handled outside the automated path.";
      ta.focus();
    });
    composer.replaceChildren(
      el("div", { class: "copilot-section", text: "Reply" }),
      ta,
      el("div", { class: "row" },
        aiBtn,
        el("button", { class: "btn", type: "button", text: "Send", onclick: () => {
          if (!ta.value.trim()) { ta.focus(); return; }
          composer.querySelector("[data-sent]")?.remove();
          composer.appendChild(el("div", { class: "small text-muted", "data-sent": "1",
            text: "Demo workspace: sending is disabled. The AI decision trail is on the right." }));
        } }),
        escalateBtn,
        el("span", { class: "grow" }),
        el("span", { class: "hint", text: "Last message: " + (msgs[msgs.length - 1]?.role || "—") })));
  }

  /* ================= RIGHT: AI Copilot ================= */

  function copilotPane() {
    return el("div", { class: "copilot-pane", id: "copilot-pane" });
  }

  function renderCopilot() {
    const pane = document.getElementById("copilot-pane");
    if (!pane) return;
    if (!state.conversation) {
      pane.replaceChildren(emptyBox("No ticket selected", "Select a conversation to see the AI Copilot."));
      return;
    }
    if (state.analysis) {
      pane.replaceChildren(...copilotContent(state.analysis));
    } else {
      pane.replaceChildren(
        card("AI Copilot",
          el("div", null,
            el("p", { class: "small text-secondary" },
              "Run the copilot to classify this customer's request, retrieve similar resolved cases, and draft an evidence-backed reply."),
            el("div", { class: "mt-4" },
              el("button", { class: "btn primary", type: "button", text: "Analyze with AI", onclick: () => runCopilot() }))),
          null));
    }
  }

  async function runCopilot() {
    if (state.analyzing || !state.conversation) return;
    const conv = state.conversation;
    // The customer's latest (or root) message is what the pipeline analyzes —
    // exactly what the real workflow would send.
    const customerMsgs = (conv.messages || []).filter((m) => m.role === "customer");
    const message = (customerMsgs[customerMsgs.length - 1] || (conv.messages || [])[0] || {}).text;
    if (!message) return;

    state.analyzing = true;
    const pane = document.getElementById("copilot-pane");
    pane.replaceChildren(card("AI Copilot",
      el("div", null,
        el("div", { class: "state-box", text: "Analyzing customer request…" }),
        el("p", { class: "hint mt-3", text: "Classifying intent → retrieving historical evidence → drafting → verifying claims → deciding." })),
      null));
    const aiBtn = document.querySelector(".convo-composer .btn.primary");
    if (aiBtn) { aiBtn.disabled = true; aiBtn.textContent = "Analyzing…"; }

    try {
      const result = await API.respond(message, 5, conv.conversation_id);
      state.analysis = result;
      pane.replaceChildren(...copilotContent(result));
    } catch (err) {
      pane.replaceChildren(card("AI Copilot", el("div", null,
        errorBox(err),
        el("button", { class: "btn mt-3", type: "button", text: "Retry", onclick: () => runCopilot() })), null));
    } finally {
      state.analyzing = false;
      if (aiBtn) { aiBtn.disabled = false; aiBtn.textContent = "AI Assist"; }
    }
  }

  function copilotContent(result) {
    return [
      decisionCard(result),
      intentCard(result),
      evidenceCard(result),
      suggestedReplyCard(result),
      whyCard(result),
    ];
  }

  /* AI decision card: AUTO or ESCALATE with the reason codes. */
  function decisionCard(result) {
    const d = result.decision || {};
    const isAuto = d.decision === "AUTO";
    const codes = d.reason_codes || [];
    const checks = isAuto
      ? ["Evidence sufficient", "Historical resolution match", "Low risk", "Grounding passed"]
      : codes.map((c) => codeShort(c));
    return el("div", { class: "decision-banner " + (isAuto ? "auto" : "escalate"), role: "status" },
      el("div", { class: "d-label", text: "AI decision" }),
      el("div", { class: "d-value", text: isAuto ? "✓ AUTO" : "⚠ ESCALATE" }),
      el("div", { class: "row wrap mt-2" },
        badge(`risk: ${d.risk_level ?? "—"}`,
          d.risk_level === "high" ? "red" : d.risk_level === "medium" ? "amber" : "green"),
        badge(`provider: ${result.provenance?.mode || "?"}`, result.provenance?.mode === "mock" ? "outline-mock" : "green")),
      el("ul", { class: "reason-list mt-3" },
        checks.map((c) => el("li", { class: isAuto ? "good" : "bad" },
          el("span", { class: "bullet", text: isAuto ? "✓" : "!" }), el("span", { text: c })))),
      el("p", { class: "small text-secondary mt-3", text: d.reason || "" }));
  }

  function codeShort(code) {
    return (REASON_STORIES[code] || code).split("—")[0].split(".")[0];
  }

  function intentCard(result) {
    const conf = result.intent?.confidence ?? 0;
    const alts = Object.entries(result.intent?.all_scores || {})
      .filter(([k]) => k !== result.intent.name).slice(0, 3);
    const amb = result.ambiguity;
    const chips = [];
    if (result.novelty?.is_ood) chips.push(badge("out-of-distribution", "red", "Message resembles no historical case"));
    if (amb?.is_ambiguous) chips.push(badge("ambiguous", "amber", "Top-2 intent margin is narrow"));
    if (amb?.multi_intent_suspected) chips.push(badge("multi-intent", "red", "Several distinct issues in one message"));
    if (amb?.retrieval_disagreement) chips.push(badge("retrieval disagreement", "amber", "History mostly points to a different intent"));

    return card("Intent",
      el("div", null,
        el("div", { class: "row between" }, intentBadge(result.intent.name),
          el("strong", { text: Math.round(conf * 100) + "%", tip: "Classifier confidence for this intent (calibrated model output, not a promise)." })),
        el("div", { class: "confidence-viz" },
          el("div", { class: "cv-scale", role: "img", "aria-label": `Confidence ${Math.round(conf * 100)} percent` },
            el("div", { class: "cv-marker", style: `left: ${(conf * 100).toFixed(1)}%`, tip: "Auto-handling needs ≥ 75% confidence" })),
          el("div", { class: "cv-labels" }, el("span", { text: "0%" }), el("span", { text: "auto ≥ 75%" }), el("span", { text: "100%" }))),
        chips.length ? el("div", { class: "row wrap mt-3" }, chips) : null,
        alts.length ? el("div", { class: "mt-4 section-title", text: "Alternatives the classifier considered" }) : null,
        alts.length ? el("div", { class: "hbar-list" },
          alts.map(([name, p]) => el("div", { class: "hbar-row" },
            el("span", { class: "name", text: name }),
            el("div", { class: "bar" }, el("span", { style: `width:${(p * 100).toFixed(1)}%` })),
            el("span", { class: "val", text: num(p) })))) : null));
  }

  function evidenceCard(result) {
    const cases = result.evidence?.cases || [];
    const ev = result.evidence || {};
    const body = el("div");

    body.appendChild(el("div", { class: "row between mb-3" },
      el("span", { class: "small text-muted",
        text: `${cases.length} similar resolved case(s) · intent agreement ${Math.round((ev.intent_agreement_rate ?? 0) * 100)}%` }),
      ev.hybrid_enabled ? badge("hybrid ranking", "gray", "Semantic + lexical + intent-compatibility + resolution-quality reranking") : null));

    if (!cases.length) {
      body.appendChild(emptyBox("No similar historical cases",
        "Without historical evidence the AI will not draft a reply — it escalates."));
      return card("Historical evidence", body);
    }

    cases.slice(0, 4).forEach((c) => {
      body.appendChild(el("div", { class: "evidence-card" },
        el("div", { class: "ev-head" },
          el("div", { class: "row" },
            el("strong", { class: "small", text: "Case " + c.conversation_id.split("_").pop() }),
            el("span", { class: "mono small text-muted", text: c.conversation_id }),
            intentBadge(c.intent)),
          el("span", { class: "badge blue", text: Math.round((c.similarity || 0) * 100) + "% similar",
            tip: "Cosine similarity between this historical message and the customer's message." })),
        el("div", { class: "ev-body" },
          el("div", { class: "quote" }, el("span", { class: "who", text: "Customer asked" }), c.customer_message),
          c.resolution
            ? el("div", { class: "quote" }, el("span", { class: "who", text: "How it was resolved" }), c.resolution)
            : el("div", { class: "hint" }, "No recorded resolution for this case."),
          el("details", null,
            el("summary", { class: "small text-muted", text: "Why this evidence?" }),
            el("div", { class: "small text-secondary mt-2", text: c.explanation || 
              "Retrieved because its customer message is textually similar to the current request" +
              (c.components && Object.keys(c.components).length
                ? ` (components: ${Object.entries(c.components).map(([k, v]) => `${k} ${num(v)}`).join(", ")})` : "") + "." })))));
    });

    body.appendChild(el("div", { class: "mt-4 section-title", text: "Evidence quality" }),
      el("div", { class: "row wrap" },
        badge(`evidence score ${num(result.evidence_score)}`,
          result.evidence_score >= 0.55 ? "green" : "amber",
          "Empirically calibrated quality score over the retrieved cases"),
        badge(`resolution relevance ${Math.round((ev.resolution_agreement_rate ?? 0) * 100)}%`,
          (ev.resolution_agreement_rate ?? 0) >= 0.5 ? "green" : "amber",
          "Share of cases whose recorded resolution is actionable")));
    return card("Historical evidence", body);
  }

  function suggestedReplyCard(result) {
    const resp = result.response || {};
    const cv = resp.claim_verification;
    const grounded = resp.grounded;
    const body = el("div");

    if (resp.is_mock) {
      body.appendChild(el("div", { class: "mock-warning mb-3" },
        el("span", { class: "icon", text: "⚠" }),
        el("span", { text: "MOCK RESPONSE — deterministic offline provider, not a real LLM. Shows the pipeline shape only." })));
    }
    if (resp.generation_status === "FAILED") {
      body.appendChild(el("div", { class: "callout danger mb-3" },
        el("div", { class: "callout-title", text: "Generation failed" }),
        el("p", { class: "small text-secondary" },
          `${resp.provider || "The provider"} could not produce a valid draft. The AI escalated instead of sending an unverified answer.`)));
      body.appendChild(emptyBox("No suggested reply", "The decision card explains why."));
      return card("Suggested reply", body);
    }
    if (!resp.draft) {
      body.appendChild(emptyBox("No suggested reply", "Generation was skipped or failed; the decision card explains why."));
      return card("Suggested reply", body);
    }

    body.appendChild(el("div", { class: "draft-quote", text: resp.draft }));
    body.appendChild(el("div", { class: "row wrap mt-3" },
      el("span", { class: "small", style: "font-weight: 600; color: " + (grounded ? "var(--green)" : "var(--red)"),
        text: grounded ? "✓ Evidence-backed" : "✗ Not evidence-backed" }),
      cv ? badge(`grounding ${Math.round((cv.score ?? 0) * 100)}%`,
        cv.passed ? "green" : "red",
        "Share of draft claims verified against retrieved historical cases") : null,
      badge(`risk: ${result.decision?.risk_level ?? "—"}`,
        result.decision?.risk_level === "high" ? "red" : result.decision?.risk_level === "medium" ? "amber" : "green")));

    if (cv && cv.claims?.length) {
      body.appendChild(el("details", { class: "mt-3" },
        el("summary", { class: "small text-muted", text: "Per-claim verification" }),
        el("div", { class: "claim-list mt-3" }, cv.claims.map((c) => {
          const ok = c.status === "supported";
          return el("div", { class: `claim ${ok ? "ok" : c.status === "unsupported" ? "bad" : "neutral"}` },
            el("div", { class: "row" },
              el("span", { class: "claim-mark", text: ok ? "✓" : c.status === "unsupported" ? "✗" : "·" }),
              el("span", { class: "claim-text", text: c.claim_text })),
            el("div", { class: "claim-meta small" },
              el("span", { class: "text-muted", text: c.explanation })));
        }))));
    }
    return card("Suggested reply", body);
  }

  /* Why did the AI decide this? — the trust feature. */
  function whyCard(result) {
    const d = result.decision || {};
    const codes = d.reason_codes || [];
    const isAuto = d.decision === "AUTO";
    const items = [];
    if (isAuto) {
      items.push(["good", "Intent confidence above the configured auto-handling threshold."],
        ["good", `${result.evidence?.cases?.length ?? 0} similar historical resolutions found.`],
        ["good", "Historical cases agree on how this request is resolved."],
        ["good", "No high-risk intent or privacy signal detected."],
        ["good", "Generated response passed grounding validation."]);
    } else {
      for (const c of codes) items.push(["bad", REASON_STORIES[c] || c]);
      if (!codes.length) items.push(["bad", "The policy escalated this conversation."]);
      items.push(["bad", "A human review is recommended."]);
    }
    return card("Why did the AI decide this?",
      el("ul", { class: "reason-list" },
        items.map(([tone, text]) => el("li", { class: tone },
          el("span", { class: "bullet", text: tone === "good" ? "✓" : "!" }),
          el("span", { text }))));
  }

  window.Pages = window.Pages || {};
  window.Pages.inbox = render;
})();
