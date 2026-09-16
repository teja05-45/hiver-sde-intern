# Technical Debt

Only real, currently-true issues. Severity = impact on the system's purpose (safe automation), not
generic code-style nitpicks.

| Issue | Severity | Impact | Proposed Fix |
|---|---|---|---|
| TF-IDF retrieval has no semantic matching (Recall@1 49.5%) | High | Weak evidence quality caps everything downstream — drafts can be grounded in irrelevant history | Sentence embeddings behind the existing `EmbeddingProvider` interface; re-measure Recall@k, then recalibrate evidence weights |
| Automation precision measured on only 32 golden auto-handled cases (95% CI 51–82%) | High | The trustworthy precision number is too wide to pin an operating point; business thresholds are being set on a fuzzy measurement | Grow the golden set to ~600 with stratified sampling of threshold-boundary cases |
| Evidence-score calibration targets classifier correctness, not groundedness | High | The score optimizes the wrong proxy; `resolution_consistency` (0.0) and `evidence_count_score` (0.016) weights are artifacts of that mismatch | Re-run calibration against groundedness labels once live-generation data exists |
| Ambiguity/multi-intent signals are direction-tested, not validated | Medium | They only escalate (safe failure direction), but may be over- or under-firing invisibly — no measured precision/recall | Evaluate against the golden set's `ambiguous`/`multi_intent` category flags; tune thresholds on dev only |
| Grounding check is literal word-overlap, not semantic entailment | Medium | Paraphrased claims can pass (false accept) or fail (false reject); measured weakness documented | Two-stage check: overlap gate + embedding/LLM entailment for borderline claims |
| `general_other` mixes genuine catch-all with ambiguity bucket | Medium | Per-intent metrics for it are uninterpretable; OOD performance is overstated | Split "off-topic/noise" from "genuinely ambiguous" in the next taxonomy revision; add a calibrated OOD score |
| Golden set had a single labeler | Medium | Label quality is unverifiable (no inter-annotator agreement); some "hard" labels are genuinely contestable | Second independent labeler on a 100-example overlap subset; report κ |
| Conversation reconstruction can duplicate customer messages across engagements | Medium | Over-counts conversations (documented in `conversation.py`); inflates index size, doesn't corrupt evidence pairing | Deduplicate by root tweet id at index build; measure index delta |
| Retrieval index is rebuilt in memory per worker | Low | ~17k vectors is cheap now; 10x corpus growth would multiply memory per gunicorn worker | Move to a shared vector store (FAISS/external service) when corpus demands |
| Live-LLM cost/latency uncharacterized | Low | No p95 latency or cost-per-ticket data for capacity planning | Benchmark live generation over a sample; add timeouts/metrics per stage |
| Frontend is no-framework vanilla JS | Low | Fine at current size (8 pages); DOM-building code would get unwieldy if it grows | Adopt a light framework only if pages multiply; keep the API-first design |
| MODEL_NAME default hardcodes a Groq model ID that can be retired | Low | A Groq-side retirement causes 404s until config changes (happened once: `llama-3.3-70b-versatile` retired; fixed to a verified live ID) | Optional: query `/models` at startup and fail fast with a clear error when the configured model is absent |
