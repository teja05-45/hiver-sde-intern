# REPORT — Evidence-First Customer Support Agent for AmazonHelp

> **NOTE: this is a stale earlier draft kept for history. The authoritative report is
> [`reports/final_report.md`](reports/final_report.md) — its numbers (19.5% golden accuracy,
> 74.1-point gap) supersede the ones below, which came from earlier golden-set passes.**

## 1. Executive Summary

This project builds an AI support agent for AmazonHelp that classifies customer intent, retrieves
grounded historical evidence, drafts a response, and decides AUTO-HANDLE vs. ESCALATE with a
stated reason. The central finding is not a model result — it's a measurement result: **the
system's silver-label headline metrics (93.6% intent accuracy) do not survive
contact with hand-verified human labels** (19.5% accuracy). That gap, how
it was found, and what it implies, is the most important content in this report. Full detail:
`reports/misleading_headline_number.md`.

## 2. Problem Framing

Given ~2.81M real, noisy Twitter customer-support exchanges across 108 brand accounts, build a
system that (a) understands what a customer wants, (b) answers only when it can point to real
historical precedent, and (c) knows when to get a human involved. The binding constraint
throughout this build was environmental, not conceptual: **zero outbound network access** in the
build sandbox (confirmed against Kaggle, PyPI, and a generic HTTPS probe). This shaped nearly
every technology choice — see `docs/decision-log.md`.

## 3. Brand Selection

`scripts/profile_dataset.py` scored 108 candidate brand accounts on a composite of log-scaled
engagement volume, average reply-thread depth, and engagement rate. **AmazonHelp** won: 168,814
direct customer engagements, 71,048 unique customers, 99.4% engagement rate, 3.81-turn average
thread depth. A first scoring draft (linear-capped volume) would have picked a brand with 15x less
data; caught and fixed before proceeding (`reports/brand_profile.md`).

## 4. Data Preparation

Conversation reconstruction walks the tweet reply-graph backward from each AmazonHelp reply to
its root customer message (capped at depth 12), fetching text for only the tweet IDs actually
needed — never the full 2.81M-row file at once. A deterministic, seeded sample of 60,000
engagements was reconstructed (`scripts/build_dataset.py`, ~30s). **Leakage prevention** used a
temporal split (train `<2017-11-10`, dev `[11-10,11-25)`, test `>=11-25`) instead of random:
measured 2,701 cross-split duplicate-content groups under a naive random split vs. 16 under the
temporal split (`reports/leakage_analysis.json`).

## 5. Intent Taxonomy

12 intents (`configs/intents.yaml`), discovered via TF-IDF+MiniBatchKMeans clustering (k=14,
selected by silhouette score) on 44,375 English-filtered customer messages, then human-merged and
split down to 12 operationally distinct categories with documented cluster provenance. Two real
bugs were caught and fixed during this phase: a non-ASCII language filter that missed
Latin-script European languages (fixed with a function-word-overlap heuristic), and an overly
generic keyword signal (`"account"`) causing false-positive labels.

## 6. Agent Architecture

```
Customer message → Intent Classifier (TF-IDF+LogReg) → Historical Retriever (TF-IDF cosine,
index built ONLY from data strictly before the query period) → Evidence Scorer (empirically
calibrated weights) → LLM Generator (evidence-only prompting, must abstain if insufficient) →
Grounding Validator (independently checks claims against evidence text) → Escalation Policy
(multi-signal, named reason codes) → AUTO or ESCALATE
```

Each stage is an independent, unit-tested module (`backend/app/{classification,retrieval,
generation,escalation}/`) — no single mega-prompt. `backend/app/services/agent.py` orchestrates
and fails safe at every stage (retrieval/generation failure → escalate, never fabricate).

## 7. Baselines

| Model | Accuracy (silver) | Macro-F1 (silver) | Accuracy (golden) | Macro-F1 (golden) |
|---|---|---|---|---|
| Majority | 23.7% | 0.032 | 15.5% | 0.022 |
| TF-IDF + LogisticRegression | 93.6% | 0.869 | **19.5%** | **0.182** |

The proposed system uses the same classifier as its intent stage — the "proposed system"
improvement over baselines is in the retrieval/evidence/escalation layers around it, not in
classification accuracy itself, which is honestly the weakest link right now (see §12).

## 8. Evaluation Methodology

Classification: accuracy, macro/weighted P/R/F1, per-intent F1, confusion matrix
(`backend/app/evaluation/metrics.py`, implemented from scratch, no hidden sklearn version
dependency). Retrieval: Recall@1/3/5, MRR, with retrieved-case-shares-query-intent as a stated
distant-supervision relevance proxy (`scripts/evaluate_retrieval.py`). Evidence scoring:
empirically calibrated via logistic regression against classifier-correctness on the dev split,
not hand-picked (`scripts/calibrate_evidence_score.py`). Automation: full precision/coverage
threshold sweep (`scripts/evaluate_automation.py`), operating point chosen by max-coverage-at-
target-precision, not by whichever point looked best.

## 9. Results

Retrieval: Recall@1=42.6%, Recall@3=66.5%, Recall@5=77.5%, MRR=0.553 — real, non-inflated (avg
top similarity only 0.30, consistent with TF-IDF's lack of semantic matching).

Automation: silver-calibrated threshold (evidence_score≥0.30, intent_confidence≥0.80) achieves
99.0% precision at 20.1% coverage on the silver test set — **but only 68.75% precision (95% CI
51.4-82.1%) at 16.0% coverage when the identical, untouched threshold is measured against the
golden set** (`reports/golden_automation_check.json`). This is the headline result of the entire
evaluation section: report both, always, side by side.

## 10. LLM Judge Validation

Harness fully implemented and unit-tested (`backend/app/evaluation/judge.py`,
`backend/app/evaluation/agreement.py` — Spearman correlation, weighted Cohen's kappa,
exact/adjacent agreement). Executed LIVE on 2026-09-16: 50 golden-subset replies generated by the
live Groq provider, all 50 judged by the live judge (0 failures), and compared against **proxy**
human scores derived programmatically from golden labels (`scripts/score_human_proxy.py`).
Agreement is weak-to-moderate (exact 4–72% per dimension, Spearman ρ ≈ 0.0–0.24) — reported as-is
in `reports/judge_human_agreement.json`. Status: **pipeline verified end-to-end against a real LLM;
NOT validated against real human judgments** — the proxy provenance is stated in the report file,
the API, and the UI, and no human-agreement claim is made anywhere.

## 11. Failure Analysis

From real golden-set misclassifications (`reports/failure_analysis.md`, `scripts/analyze_failures.py`):
97/200 (48.5%) failures, top categories: ambiguous intent (36.1% of failures — classifier's own
top-2 margin <0.15), out-of-distribution (19.6% — doesn't cleanly fit any of the 12 intents),
other classifier error (18.6%), multi-intent messages (12.4%), noisy short messages (7.2%). Every
example is a real misclassification with expected/predicted intent and confidence shown, not an
invented illustration.

## 12. What Is Misleading About My Headline Number?

Full document: `reports/misleading_headline_number.md`. Summary: silver-label accuracy (88.7%) and
silver-calibrated auto-precision (99%) are both real numbers from real code, and both substantially
overstate real-world performance because the labels being evaluated against were produced by a
closely related process to the thing being evaluated (TF-IDF clustering → TF-IDF classification).
The honest numbers, from 200 independently hand-verified examples: **~52% intent accuracy, and
auto-precision that has not been shown to exceed ~80% at any statistically reliable sample size.**

## 13. Limitations

Live LLM: the provider path is verified (health check, 50 golden-subset generations, 50 live judge
calls — FINAL_VERIFICATION.md §5b/§5c), but generation quality at scale over the full golden set and
the judge's agreement with REAL humans remain unmeasured (current agreement numbers are judge-vs-
proxy). TF-IDF has no semantic matching (retrieval ceiling). Single primary labeler for the golden
set (no independent second annotator for true inter-annotator agreement). 200 golden examples is
enough to see large effects but too few to precisely pin down a specific automation operating point.
Docker images were built and smoke-run locally (gunicorn boot + health + respond verified); not
load-tested or production-hardened.

## 14. What I Would Do With One More Week

1. Scale the live LLM evaluation: generation over the FULL golden set (only the 50-example judge
   subset has live generation so far), and judge comparison against REAL human scores — the current
   agreement numbers are judge-vs-proxy and labeled as such.
2. Expand the golden set past 200, specifically to tighten the automation-precision confidence
   interval enough to trust a specific deployment threshold.
3. Real sentence embeddings for retrieval/clustering (biggest single lever on evidence quality —
   Recall@1 of 42.6% bottlenecks everything downstream).
4. A second independent human labeler on the golden set for genuine inter-annotator agreement.
5. Re-derive `resolution_consistency`/`evidence_count_score` weights once real generation output
   exists to calibrate against the right target (generation trustworthiness, not classifier
   correctness).

## 15. Decision Log

15 non-obvious decisions with reasons and trade-offs: `docs/decision-log.md`.
