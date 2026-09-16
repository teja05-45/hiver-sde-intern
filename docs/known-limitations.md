# Known Limitations

Stated, not hidden. Each limitation is real, currently true, and — where a
fix exists — scoped with its cost.

## Model & data

1. **Golden accuracy is 19.5% — and that is the finding.** The 2026-09-12
   cluster→intent remap (decision #19) left specific intents with as few as 6
   training examples; the classifier scores 93.6% against silver labels but
   19.5% against 200 human-verified examples. Retraining from the pre-remap
   mapping is the real fix; it was deliberately NOT applied silently because
   it rewrites every committed report. The system's answer to weak
   classification is abstention, which is why escalation behavior — not
   accuracy — is the primary safety metric.
2. **TF-IDF has no semantic matching.** Recall@1 = 49.5% bounds evidence
   quality for everything downstream. Sentence embeddings behind the existing
   `EmbeddingProvider` interface are the identified fix (~1 file swap).
3. **`general_other` is an uncertainty bucket**, not a business intent. It
   mixes promotional/off-topic noise with genuinely ambiguous messages, so
   its per-intent metrics are uninterpretable.
4. **Silver labels are circular.** They come from the same clustering process
   that shaped the taxonomy/training. Silver numbers are only reported
   side-by-side with golden numbers, never alone.

## Signals & policy

5. **Ambiguity, multi-intent, and OOD signals are experimental.** They are
   direction-tested in unit tests and against the 10-message probe suite, and
   the multi-intent detector matches the golden set's own category flags
   exactly (TP=41, FP=0, FN=0), but the ambiguity and OOD signals have no
   labeled ground truth of their own. They are ESCALATE-only: they can veto
   an AUTO, never force one — so their failure direction is safe.
6. **Grounding verification is literal, not semantic.** Claims verify by
   stemmed word-overlap and verbatim-span detection against the evidence
   text. Close paraphrases can false-reject (mitigated by distinguishing
   assertions from suggestions) and fluent fabrications can false-accept
   (mitigated by the 0.6 assertion threshold and evidence-union boundary).
   Semantic entailment would need an embedding/LLM check.
7. **Evidence-score calibration targets the wrong proxy.** Weights were fitted
   against classifier correctness, not generation groundedness;
   `resolution_consistency` carries ~0 weight as an artifact of that mismatch
   (decision #7). Recalibration needs live groundedness labels.

## Evaluation

8. **The golden set is 200 examples with one labeler.** The automation-
   precision confidence interval (±15 points, n=32 auto-handled cases) is too
   wide to pin a deployment threshold. No inter-annotator agreement was
   measured; a second independent labeler is the fix.
9. **The golden set is deliberately hard-weighted** (118/200 hard, 41
   multi-intent, 37 ambiguous), so golden accuracy is NOT expected real-
   traffic accuracy — it is a stress-test of the safety layer.
10. **The LLM judge is NOT VALIDATED.** The harness ran live (50/50 judged
    via Groq), but the comparison scores are programmatic proxies derived
    from golden labels — not human ratings. The UI states this wherever the
    agreement numbers appear.
11. **Temporal narrowness.** All data is Oct–Dec 2017 Twitter support;
    language, policies, and channel norms drift.

## Operational

12. **Docker is smoke-verified only** — images build, gunicorn boots, health
    and respond endpoints answer through the nginx proxy. Not load-tested,
    TLS-terminated, or horizontally scaled.
13. **Rate limiting is per-process in-memory.** Fine for a single-container
    demo; multi-worker deployments would need a shared store.
14. **No persistent decision store.** Request IDs correlate API responses to
    structured logs, but there is no database of past decisions to query.
15. **Provider state is measured per check, not continuously monitored.** The
    UI verifies on load; production would add periodic health probes and
    alerting.
16. **Mock mode is visible, not prevented, in production config.** A missing
    API key in `APP_ENV=production` falls back to mock with visible MOCK
    labels everywhere rather than failing startup — fail-fast would be
    stricter and is recorded in TECHNICAL_DEBT.md.
