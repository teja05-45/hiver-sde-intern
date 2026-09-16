# Golden Evaluation Set — Methodology and Limitations

**File:** `golden_set.csv` | **Size:** 200 examples | **Brand:** AmazonHelp

## How examples were sampled

The pool was the **test split only** (7,560 examples, all from the period `>= 2017-11-25`,
strictly held out from everything trained/indexed on). Using it again for the golden set
introduces no new leakage — nothing downstream trains on it.

Examples were drawn via **stratified sampling across 8 categories**, each computed from a real
signal already in the pipeline (not arbitrary):

| Category | Definition |
|---|---|
| `common` | intent is one of the 4 highest-support intents in train |
| `rare` | intent is one of the 3 lowest-support intents in train |
| `medium` | everything else |
| `ambiguous` | classifier's top-2 predicted probabilities differ by < 0.15 |
| `multi_intent` | message matches `positive_signals` keywords from 2+ different intents |
| `noisy` | message is very short (<6 tokens) or has unusual punctuation/caps density |
| `ood` | predicted intent is `general_other`, or top retrieval similarity is in the bottom 10% |
| `high_risk` | intent's `escalation_tendency` is `high` |

A single example can belong to multiple categories. Roughly ~22 examples were drawn per
category (deduplicated), topped up to 200 from the general pool. **This means the golden set is
deliberately weighted toward hard/ambiguous/rare/OOD cases** — only 43/200 (21.5%) are "easy."
That's intentional per the assignment's stratification requirement, but it also means **golden-set
accuracy should not be read as "expected accuracy on typical traffic"** — see
`reports/misleading_headline_number.md`.

Resulting distribution: `difficulty` = 118 hard / 39 medium / 43 easy. `expected_escalation` =
119 True / 81 False. Full intent distribution in `golden_set_summary.json`.

## How examples were labeled — and a finding worth being direct about

The build script (`scripts/build_golden_set.py`) first produces a *draft* label per example: a
keyword rule (checking `configs/intents.yaml`'s `positive_signals`, requiring 2+ keyword hits to
trust the rule over the model) if one fires with enough confidence, otherwise falling back to the
TF-IDF+LogisticRegression classifier's own prediction.

That fallback is a real methodological trap: **evaluating a classifier against labels that are
substantially just that classifier's own predictions is circular** and would produce a fake
near-100% "accuracy." Only 9 of the 200 draft labels came from the confident keyword rule; the
other 191 came from the classifier fallback — which meant the initial draft file was not fit to
evaluate against as-is.

So a full manual review pass was done: **every one of the 200 messages was read individually
by the primary labeler against the taxonomy definitions in `configs/intents.yaml`**, and
corrected where the draft label looked wrong. The corrections file proposed 94 changes; **86
labels actually changed value** (8 proposals matched the draft already) — `label_source =
manual_primary_labeler_review` marks exactly these 86 rows; the original draft label is
preserved in `labeler_notes`.

Read that 86/200 (43%) figure carefully: it is **a relabel rate on a deliberately hard-stratified sample**,
not a claim that the classifier is wrong on half of all traffic. Many of the changes were
genuinely close calls (e.g. `return_or_replacement_request` vs. `delivery_not_received` vs.
`general_other` for a message about a damaged item that also never arrived) where a different
careful reader could reasonably land elsewhere. It is real signal that this is a harder labeling
problem than the taxonomy's clean category names suggest — exactly the kind of thing worth
surfacing, not smoothing over.

## Single-labeler limitation, stated directly

**There was one primary labeler (Claude), not an independent multi-annotator panel.** The
assignment's guidance for this situation is to use "adjudication/re-review rather than pretending
to have multiple independent annotators" — that's what happened here (the manual review pass
described above *is* the adjudication step, applied against the classifier's draft as the second
"annotator" being checked, not against a second independent human).

A genuinely independent human-labeled subset (for measuring real inter-annotator agreement, not
just labeler-vs-classifier agreement) is a documented next step — see the human-review workflow
noted in the main README.

## What "required_evidence" and "escalation_reason" mean

`required_evidence` is populated from the assigned intent's taxonomy description — the kind of
historical evidence that would be needed to answer this message safely. `escalation_reason` is
populated when the example is expected to escalate, explaining why (high-risk intent / OOD /
ambiguous), for use in evaluating the escalation policy's own stated reasons against expectation.

## Limitations

- Single primary labeler (see above).
- Language filter (`app/services/lang_id.py`) is a lightweight function-word heuristic, not a
  real language-ID model — some non-English messages may have slipped through if the customer
  code-switched mid-message.
- Multi-turn context beyond the first 1-2 customer turns is truncated in the `context` field for
  readability; full conversations are in `data/processed/conversations_AmazonHelp.jsonl`.
- 200 examples is enough to see large effects (like the 37-point silver-vs-golden accuracy gap)
  but is genuinely too small to precisely pin down metrics like automation precision at a specific
  operating point — see the wide confidence interval in `reports/golden_automation_check.json`.
