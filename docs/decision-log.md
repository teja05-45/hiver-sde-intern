# Decision Log

19 non-obvious engineering decisions made during this project. Each includes the reason and the
trade-off accepted.

### 1. Brand selection: log-scale volume scoring, not linear-capped

**Decision:** Score candidate brands with `log1p(engagements) / log1p(max_engagements)` for the
volume component, not a linearly-capped score.
**Reason:** A first draft capped volume scoring at 5,000 engagements, which let a brand with
11K engagements (`MicrosoftHelps`) outrank one with 169K (`AmazonHelp`) purely on slightly deeper
reply chains. That's the wrong trade-off when the downstream golden set needs breadth across rare
intents.
**Trade-off:** Log scaling still lets depth/engagement-rate meaningfully move the ranking for
brands of similar volume — it doesn't just crown whoever has the most data.

### 2. Temporal split instead of random split

**Decision:** train/dev/test split by date (`< 2017-11-10` / `[2017-11-10, 2017-11-25)` /
`>= 2017-11-25`), not `train_test_split(random_state=42)`.
**Reason:** Measured, not assumed: a naive random split would put 2,701 duplicate-content groups
(7,365 documents) across train/test; the temporal split has 16 groups (65 documents) — see
`reports/leakage_analysis.json`. Deployment only ever sees future messages; temporal split is the
closer proxy.
**Trade-off:** Smaller effective test set for hyperparameter work (can't shuffle in extra data
from other periods), and results reflect Oct-Dec 2017 specifically.

### 3. TF-IDF instead of neural sentence embeddings

**Decision:** All embedding/clustering/retrieval uses TF-IDF + TruncatedSVD, behind an
`EmbeddingProvider` interface.
**Reason:** No network access in this build sandbox to `pip install sentence-transformers`.
**Trade-off:** No semantic/paraphrase matching (Recall@1 of 42.6% reflects this ceiling). The
interface isolates the choice so swapping in a real embedding model elsewhere is a ~10-line change,
not a rewrite.

### 4. Cluster-derived ("silver") labels for training, hand-labeled golden set for evaluation

**Decision:** Train the classifier on labels assigned by mapping clusters to taxonomy intents;
evaluate the headline claim against a separately hand-labeled 200-example golden set.
**Reason:** Hand-labeling tens of thousands of training examples isn't feasible; but evaluating a
classifier purely against labels its own clustering process produced is circular (see
`reports/misleading_headline_number.md`, Finding 1) — measured a 37-point accuracy gap between the
two.
**Trade-off:** Golden-set numbers have wide confidence intervals (small N); silver-set numbers are
precise but optimistic. Both are reported, never just one.

### 5. Keyword-rule threshold requires 2+ hits, not 1

**Decision:** In label assignment / golden-set drafting, a keyword rule only overrides the
classifier's prediction with 2+ `positive_signals` matches, not 1.
**Reason:** A 1-hit rule let the single generic word "account" (in `account_access_issue`'s
signal list) misfire on unrelated delivery/complaint messages that merely mentioned "my account"
in passing — caught during manual spot-check of the golden set, fixed at the source (tightened
`configs/intents.yaml`'s signal list too).
**Trade-off:** Fewer examples get the (more auditable) keyword-rule label; most fall back to the
classifier, whose own error rate is a separate, measured problem (Finding 1).

### 6. Evidence score excludes the classifier's own confidence

**Decision:** `evidence_score` is calibrated only from retrieval-based signals (similarity,
intent-agreement rate, resolution consistency, evidence count) — NOT the classifier's
`intent_confidence`, even though including it was tried first.
**Reason:** Including `intent_confidence` in the calibration captured ~88% of the fitted weight,
making `evidence_score` nearly degenerate into "restate the classifier's self-reported
confidence" — the exact "answer because the model is confident" anti-pattern the assignment's core
design principle rejects. `intent_confidence` is still used, but as its own separate, named input
to the escalation policy.
**Trade-off:** The resulting evidence-only calibration is weaker (AUC 0.60 vs 0.81) at predicting
classifier correctness — an honest cost of keeping the signal conceptually clean.

### 7. `resolution_consistency` and `evidence_count_score` calibrated to ~0 weight

**Decision:** Accept the empirical calibration result (near-zero weight for these two features)
rather than manually asserting a "should matter" weight.
**Reason:** Their true purpose (predicting *generation trustworthiness*) isn't the same target the
calibration data could measure (*classifier correctness*) — a real proxy-target mismatch, not
evidence the features are useless. Documented rather than patched with an arbitrary floor weight.
**Trade-off:** The evidence score is currently driven almost entirely by `retrieval_score` and
`intent_agreement_rate`; revisiting this once real generation output exists (with real
groundedness labels) is a "one more week" item.

### 8. RAG over fine-tuning

**Decision:** Ground responses via retrieval + prompting, not by fine-tuning a generation model.
**Reason:** No compute/network for fine-tuning in this sandbox, and RAG is more auditable (you can
point at exactly which historical case supported which claim) and cheaper to update (new resolved
conversations improve answers without retraining).
**Trade-off:** Response quality is bounded by retrieval quality (Recall@1=42.6%), and by
prompt-following reliability of whatever LLM is configured.

### 9. Historical agent replies are "evidence," not "current policy"

**Decision:** The generation prompt frames retrieved resolutions as a record of how similar issues
were *historically* handled, explicitly not a guarantee of current policy, and instructs the model
not to state historical behavior as a firm promise.
**Reason:** A 2017 agent reply about a refund policy may not reflect 2026 policy. Treating history
as ground truth risks confidently wrong (and potentially costly) promises.
**Trade-off:** Slightly more hedged-sounding draft replies; judged worth it for safety.

### 10. Explicit abstention over `if confidence > threshold`

**Decision:** Escalation is a multi-signal policy (`backend/app/escalation/policy.py`) with named,
inspectable reason codes, not a single confidence threshold.
**Reason:** The assignment's core thesis. A single threshold can't distinguish "low confidence
because genuinely ambiguous" from "high confidence but wrong" from "no supporting evidence" — each
needs to surface a different reason to a human reviewer.
**Trade-off:** More moving parts to calibrate and explain (mitigated by keeping every signal
independently named rather than blended into one score).

### 11. High-risk intents always escalate regardless of evidence strength

**Decision:** `account_access_issue`, `payment_or_billing_issue`, `cancellation_or_refund_request`,
and `customer_service_complaint` are hard-gated to ESCALATE even with perfect evidence scores.
**Reason:** Business-rule ceiling: financial/security actions and complaints-with-no-clear-fix
carry asymmetric downside if auto-handled wrong, independent of how strong the statistical
evidence looks.
**Trade-off:** Caps automation coverage — verified in the automation curve that this materially
limits how high coverage can go even at low precision bars.

### 12. Macro-F1 reported alongside accuracy, always

**Decision:** Every classification report includes macro-F1 and per-intent F1, not just accuracy.
**Reason:** `general_other` alone is 28% of training data; a model that's great at the 4 common
intents and bad at rare ones can still post high accuracy. Macro-F1 (0.486 on golden, vs 0.515
accuracy) tells a more honest story about intent-level reliability.
**Trade-off:** None really — this one's close to free, just requires remembering to report it.

### 13. Flask instead of FastAPI/Pydantic for the executed backend

**Decision:** The backend actually run and tested in this sandbox uses Flask + hand-written
validation, not FastAPI + Pydantic (the assignment's suggested stack).
**Reason:** No network access to `pip install fastapi pydantic` in this sandbox. The alternative —
writing FastAPI/Pydantic code I could never execute or test myself — would mean claiming a verified
backend that was never actually verified.
**Trade-off:** Less automatic OpenAPI doc generation and request-schema validation ergonomics than
Pydantic would give; every endpoint's schema is instead documented in its own docstring.

### 14. Mock LLM mode is the only mode exercised end-to-end in this build

**Decision:** All reported evaluation numbers, the demo, and the full test suite run against
`MockLLMProvider`, never a live Groq/Gemini call.
**Reason:** No outbound network access at all in this sandbox (confirmed: Kaggle, PyPI, and a
generic HTTPS check were all blocked with `x-deny-reason: host_not_allowed`) — there is no
credential-based workaround for a sandbox-level network restriction.
**Trade-off:** Generation quality, grounding-in-practice, and LLM-judge/human-agreement numbers are
structurally implemented and unit-tested but not measured with a real LLM. This is the single
biggest open item — see README "LLM execution status" for exactly what to run to close it.
**Result:** (2026-09-16, updated after the build) The live path has now been exercised end-to-end
outside the original sandbox: measured Groq provider health, 50 golden-subset generations, and a
50/50 live judge run vs. proxy-human scores (see FINAL_VERIFICATION.md §5b/§5c). The judge-vs-proxy
agreement numbers are reported with their provenance and are NOT presented as human validation.
Live generation over the full golden set and real-human judge comparison remain NOT RUN.

### 15. Golden set is read and re-labeled by the primary labeler, not accepted from the classifier

**Decision:** After discovering the classifier-fallback circularity (decision 4 / Finding 1),
every one of the 200 golden messages was individually read and manually confirmed or corrected,
rather than trusting the automated draft.
**Reason:** Shipping a "hand-labeled" golden set that was actually 99% classifier output would be
exactly the kind of fabricated-looking-legitimate result the assignment explicitly prohibits.
**Trade-off:** Single-labeler process (documented limitation in `data/golden/README.md`) — not a
substitute for genuine multi-annotator agreement, which would need a second independent human.

### 16. Fix `resolution_consistency` at runtime, keep the near-zero calibrated weight

**Decision:** The runtime agent now computes `resolution_consistency` (it previously passed
`resolution_embedder=None`, making the feature mechanically 0.0 on every request), while keeping
the calibrated weight of 0.0 from `configs/evidence_score_weights.json`.
**Reason:** Two distinct problems were being conflated: (a) at *calibration* time the feature's
coefficient was negative against the classifier-correctness proxy and was clipped to 0 — a
proxy-target mismatch, not proof the feature is useless; (b) at *runtime* the feature could never
fire at all because no embedder was wired. Fixing (b) costs nothing and makes the feature honest;
deleting it would destroy the ability to recalibrate against the right target (groundedness) later.
**Trade-off:** A TF-IDF embedder is fitted on the fly over retrieved resolutions per request
(small cost, bounded by k≤20); the score is currently unchanged because the weight remains 0.

### 17. Ambiguity and multi-intent signals as ESCALATE-only vetoes

**Decision:** Added explicit signals — top-2 classifier probability margin (< 0.15, the same
definition the golden set used for its `ambiguous` category), multi-intent keyword-signal
disagreement, sparse evidence, retrieval-intent disagreement, and message noise — consumed by the
escalation policy as vetoes that can turn AUTO into ESCALATE, never the reverse.
**Reason:** The failure analysis shows ambiguous (23.7%) and multi-intent (15.1%) messages are the
largest failure categories, and confidence does not catch them (golden_104: wrong at 0.986
confidence). "MULTI-INTENT → ESCALATE" beats confidently resolving one issue of several.
**Trade-off:** Coverage drops further (every vetoed message needs a human). The signals are
labeled EXPERIMENTAL: direction-tested in unit tests, not yet validated against a labeled ground
truth — and they are the only policy inputs that cannot raise risk internally without escalating
publicly.

### 18. One error shape, request IDs, and PII-free structured logs

**Decision:** Every API error returns `{"error": {code, message, request_id}}`; every request gets
an `X-Request-ID` (honoring an inbound one); every agent decision emits one JSON log line with
intent/confidence/evidence/grounding/decision/reason-codes and never the message body.
**Reason:** An auditable-by-design system needs correlatable logs and predictable errors; logging
message bodies would create a PII surface the logs don't need.
**Trade-off:** Debugging a specific failing message requires finding it via request ID in the
application database (which this demo doesn't have), not from logs alone.

### 19. Cluster→intent remap (2026-09-12): prefer defensible labels over corpus coverage

**Decision:** On 2026-09-12 the cluster→intent mapping in `configs/intents.yaml` was revised
(decision recorded in the YAML header; not previously in this log — an omission this entry
fixes). The largest text cluster (cluster 14, ~52% of the corpus) was remapped to `general_other`,
on the judgment that its members were too topically heterogeneous to justify single-intent
labels. The labeled dataset was rebuilt and the classifier retrained on 2026-09-14.
**Why:** Label provenance matters more than a flattering number. Forcing half the corpus into
specific intents produces confident-looking silver labels that don't survive human review — the
golden set (see `reports/golden_gap_analysis.md`) agrees with the post-remap silver labels only
~21.5% of the time, and golden accuracy fell from 53.5% (pre-remap artifact) to 19.5%.
**Trade-off:** Silver accuracy looks catastrophic on specific intents (`account_access_issue`
retained only 6 training examples); corpus coverage claims shrank. We report the 19.5% rather
than keeping the stale 53.5% artifact, because an honest gap is a diagnostic signal and a stale
flattering artifact is a liability.
**Important:** Do NOT cite the pre-remap 53.5% as the current golden accuracy. The current,
reproducible number is 19.5% (`reports/golden_set_evaluation.json`, regenerated 2026-09-14).
