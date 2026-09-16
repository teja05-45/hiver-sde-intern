/* API client — every piece of data the UI shows comes from the backend.
 * No metric is ever hardcoded in the frontend. */
(function () {
  "use strict";

  class ApiError extends Error {
    constructor(code, message, requestId, status) {
      super(message);
      this.code = code;
      this.requestId = requestId;
      this.status = status;
    }
  }

  async function request(method, path, body) {
    let resp;
    try {
      resp = await fetch(path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (networkErr) {
      throw new ApiError("NETWORK_ERROR", "Backend is unreachable. Is it running?", null, 0);
    }
    let data = null;
    try { data = await resp.json(); } catch (_) { /* non-JSON */ }
    if (!resp.ok) {
      const err = (data && data.error) || {};
      throw new ApiError(
        err.code || "HTTP_" + resp.status,
        err.message || `Request failed (${resp.status})`,
        err.request_id || resp.headers.get("X-Request-ID"),
        resp.status
      );
    }
    return data;
  }

  window.API = {
    ApiError,

    health: () => request("GET", "/health"),
    intents: () => request("GET", "/api/v1/intents"),

    classify: (message) => request("POST", "/api/v1/agent/classify", { message }),
    retrieve: (message, k) => request("POST", "/api/v1/agent/retrieve", { message, k }),
    respond: (message, k) => request("POST", "/api/v1/agent/respond", { message, k }),

    evaluationSummary: () => request("GET", "/api/v1/evaluation/summary"),
    failures: () => request("GET", "/api/v1/evaluation/failures"),
    automation: () => request("GET", "/api/v1/evaluation/automation"),
    intentMetrics: () => request("GET", "/api/v1/evaluation/intents"),

    goldenSummary: () => request("GET", "/api/v1/golden-set/summary"),
    goldenExamples: (params) => {
      const qs = new URLSearchParams();
      if (params.outcome) qs.set("outcome", params.outcome);
      if (params.intent) qs.set("intent", params.intent);
      if (params.q) qs.set("q", params.q);
      const suffix = qs.toString() ? "?" + qs.toString() : "";
      return request("GET", "/api/v1/golden-set/examples" + suffix);
    },

    judgeSummary: () => request("GET", "/api/v1/llm-judge/summary"),
    decisions: () => request("GET", "/api/v1/decisions"),
    system: () => request("GET", "/api/v1/system"),
  };
})();
