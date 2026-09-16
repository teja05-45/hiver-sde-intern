# Interview Guide

How to explain this project — accurately — at increasing depth.

## 30-second explanation

> I built a support agent that decides **when it's safe to answer**, not just what to answer. It
> classifies intent, retrieves how similar issues were historically resolved, drafts a reply grounded
> only in that evidence, verifies every claim, and escalates to a human when evidence is weak — with
> a named reason every time. The most important thing I found: my initial 94% headline accuracy
> collapsed against human-verified labels — to 53.5% pre-remap, and 19.5% after a taxonomy remap
> halved the specific-intent training data — and the whole system is designed around that lesson.

## 2-minute explanation

- **Data:** 2.81M real Twitter support exchanges; picked AmazonHelp via a scored brand profile
  (volume, reply depth, engagement) — not fame.
- **Taxonomy:** 12 intents discovered by TF-IDF+KMeans clustering, then human-merged/split.
- **Honesty layer:** temporal train/dev/test split (leakage measured: random split leaks 2,701
  duplicate groups vs 16 temporal), plus a 200-example hand-verified golden set.
- **The finding (current):** silver-label accuracy 93.6%; golden accuracy 19.5%. The silver labels
  came from the same clustering that shaped training — circular. (A 2026-09-12 cluster→intent
  remap — decision #19 in the decision log — cut specific-intent training data hard; see
  `reports/golden_gap_analysis.md` for the full decomposition. The pre-remap artifacts showed
  53.5%/68.75%; they are stale and no longer quoted as current.)
- **Design consequence:** the agent optimizes *safe* automation. Multi-signal escalation with reason
  codes; high-risk intents (billing, account security) always escalate; claims verified against
  evidence before anything is sent; everything is auditable in a UI built for that purpose.

## 5-minute architecture explanation

Walk the pipeline (and draw it):

1. **Intent classifier** — TF-IDF (1–2 grams) + multinomial LogisticRegression, class-weighted.
   Output: intent + full probability distribution (the distribution matters — ambiguity detection
   uses the top-2 margin).
2. **Retriever** — TF-IDF cosine over ~17k resolved conversations. Strict temporal rule: the index
   only ever contains conversations *earlier* than the evaluation period. Recall@1 ≈ 49.5% is the
   honest bottleneck.
3. **Evidence scorer** — combines top similarity, intent agreement among retrieved cases,
   resolution consistency, and evidence count. Weights are *fitted* (logistic regression on dev
   against "was the classifier right?"), not hand-picked. Classifier confidence deliberately
   excluded (it captured ~88% of weight and made the score degenerate).
4. **Generator** — LLM receives a structured JSON prompt (system policy / customer message /
   evidence as separate labeled fields — a prompt-injection defense), must answer only from
   evidence, must abstain if insufficient, returns strict JSON.
5. **Grounding validator** — checks the model's claimed grounded statements against the actual
   evidence text (word-overlap); an LLM self-report is never trusted.
6. **Escalation policy** — hard fail-safes first (retrieval/generation failure, unsupported claims,
   high-risk intent → ESCALATE), then evidence/intent/grounding thresholds, then experimental
   ambiguity and multi-intent vetoes (ESCALATE-only). Every decision carries machine-readable
   reason codes and a human-readable reason.
7. **API + UI** — Flask + gunicorn, request IDs, structured logs (one PII-free line per decision),
   and an operations console whose every number comes from the API — nothing hardcoded.

Then close with the evaluation story: baselines → retrieval metrics → automation precision/coverage
→ golden check → failure taxonomy → (planned) judge validation. The system is measured at every
layer, and the measurements changed the design.

## Questions a senior interviewer may ask

**Why TF-IDF? Why not embeddings?**
Environment constraint (no package installs in the build sandbox) plus honesty: TF-IDF is a real,
competitive classical baseline. Its weakness — no paraphrase matching — is measured, not hidden
(Recall@1 49.5%), and the `EmbeddingProvider` interface isolates the swap to ~one file. I'd rather
ship a measured TF-IDF than an untested embedding stack.

**Why a temporal split?**
Deployment only ever sees *future* messages. I measured the alternative: a random split leaks 2,701
duplicate-content groups into train/test (templated complaints); temporal, 16. Random-split accuracy
would be inflated by memorization.

**How did you prevent leakage?**
Three layers: temporal cutoffs; structural (retrieval index built only from earlier splits);
detection (normalized-content-hash duplicate analysis reported in `leakage_analysis.json`, not
silently dropped — I measure and report overlap rather than laundering it).

**Why a golden set? Why is golden accuracy so low?**
Because silver labels are circular — the classifier is graded by the process that taught it. The
golden set is 200 examples, stratified toward hard cases (ambiguous/multi-intent/OOD/noisy), each
read and corrected by hand; 43% of machine-drafted labels changed (86 of 200). Low golden accuracy is *the
finding*, not a bug. It's also deliberately hard-weighted, so it's not "expected accuracy on
average traffic."

**Why not optimize accuracy?**
Accuracy on imbalanced, safety-critical support traffic is the wrong objective. What matters is
precision on what you automate and honest escalation elsewhere. Macro-F1, per-intent F1, and
automation precision/coverage are the real dashboards.

**Why escalate at all?** / **What happens when retrieval fails?**
Asymmetric costs: a wrong automated answer to a billing dispute costs more than a routed ticket.
Failure paths escalate deterministically — retrieval failure, generation failure, unsupported claims
all short-circuit to ESCALATE with HIGH risk, before any threshold logic.

**How do you detect hallucinations? How does grounding work?**
The generator must return structured claims; the validator independently checks each claim's words
against the retrieved evidence text (≥60% overlap to count as supported), and any self-reported
unsupported claim fails the whole response. It's intentionally literal — semantic entailment would
need an embedding/LLM check, which is documented as a limitation.

**What happens when the LLM returns malformed output?**
`parse_generation_output` catches it, returns a parse_error, and the policy treats it as
generation_failed → ESCALATE. Unit-tested for both providers.

**How would you improve Recall@1?**
Sentence embeddings (measured ceiling), then query expansion with conversation context (the failure
analysis shows short/noisy root messages), then re-ranking with cross-encoders if volume justifies it.

**Why is resolution_consistency zero?**
Two separate reasons I distinguish carefully: at calibration, its coefficient was *negative* against
the classifier-correctness target (clipped to 0) — a proxy-target mismatch, not proof of uselessness;
and at runtime it was computed as 0 because no embedder was passed (a real bug I found and fixed —
it's now computed from retrieved resolutions). Re-calibrating against a groundedness target is
planned once live generation data exists.

**How would you deploy this?**
gunicorn behind a load balancer in containers (verified build), artifacts baked into the image,
`APP_ENV=production`, explicit CORS origins, secrets via env, healthchecks for orchestration, and a
staged rollout: shadow mode (log decisions, don't send) → auto-handle only low-risk intents → expand.

**How would you monitor it?**
Per-decision structured logs (request_id, intent, confidence, evidence score, grounding, decision,
reason codes) → dashboards for escalation rate, reason-code distribution, precision sampled by
weekly human audits, drift in intent distribution and retrieval similarity. The reason codes are the
key: they make escalation *debuggable*.

**How would you handle prompt injection?**
Evidence is untrusted customer text: it enters as JSON data in a labeled field, never concatenated
into instructions; the system prompt states evidence is data, not instructions; grounding validation
catches smuggled claims; high-risk outputs escalate regardless. Deterministic mock behavior confirms
the plumbing (it template-fills rather than executing evidence text).

**How would you support multiple brands?**
Everything is already brand-parameterized (artifacts, taxonomy, thresholds are per-brand files). Add
a brand registry + per-brand routing; the brand-selection script generalizes to re-profiling new
brands. The real cost is golden sets per brand — labeling, not code.

**How would you handle multi-intent?**
Currently: detect via keyword-signal disagreement across intents → ESCALATE (better than confidently
picking one). Next: multi-label classification and primary/secondary intent extraction; full
resolution-splitting is out of scope and not claimed.

**How would you detect OOD?**
Signals exist (top-2 margin, retrieval similarity, retrieval-intent disagreement) but are
experimental — direction-tested, not validated. A real solution needs calibrated OOD scoring
(energy/Mahalanobis on embeddings) with its own labeled eval.

**How would you validate an LLM judge?**
Same task, two scorers: ~50 golden responses scored by the judge (live LLM) and by a human, then
Spearman ρ per dimension, weighted Cohen's κ, exact/adjacent agreement. Until that's run, the UI
shows NOT VALIDATED and no judge-derived quality number is displayed anywhere.

**What would you build next week?**
Live-LLM evaluation end-to-end (the biggest gap), golden set to ~600 to tighten the auto-precision
CI, embedding retrieval behind the existing interface, and validation of the ambiguity signals
against the golden set's category flags.
