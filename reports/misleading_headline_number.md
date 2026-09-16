# What Is Misleading About My Headline Number?

## The headline you might reach for

> "88.7% intent classification accuracy. 99.0% precision on auto-handled responses at 20.1%
> coverage."

Both numbers are real — they came out of actual code run on actual data (see
`reports/baseline_results.json` and `reports/automation_curve.json`). **Neither should be your
takeaway number without reading this page first.**

## Finding 1: the 88.7% accuracy is measured against labels the same pipeline produced

Training and test labels for the classifier are **cluster-derived ("silver") labels**: the intent
taxonomy was discovered by TF-IDF + KMeans clustering, and every conversation's label is that
cluster's assigned intent name. The classifier being evaluated is *also* TF-IDF-based. Evaluating
a TF-IDF classifier against TF-IDF-cluster-derived labels has a structural advantage baked in —
it's partially measuring "can this model reconstruct the process that generated its own labels,"
not "does this model understand customer intent."

**The real check:** 200 examples were independently read and hand-labeled (`data/golden/`).
Against that human-verified set:

| Evaluation | N | Accuracy | Macro-F1 |
|---|---|---|---|
| Silver-label test set | 7,560 | **88.7%** | 0.870 |
| Golden set (human-verified, hard-stratified) | 200 | **51.5%** | 0.486 |
| Golden set — easy subset only | 43 | **58.1%** | 0.381 |

Even on the golden set's *easy* examples (excluding deliberately-ambiguous/multi-intent/OOD/rare
cases), accuracy is 58%, nowhere near 88.7%. The gap is not explained away by "the golden set is
just harder" — it's mainly the label circularity above. **51.5% (golden, human-verified) is the
number to trust for real-world intent accuracy on this system**, not 88.7%.

## Finding 2: 47% of golden-set draft labels needed manual correction

Building the golden set surfaced this directly: the classifier's own predictions, used as draft
labels, were wrong or debatable often enough that 94/200 (47%) were changed on manual review (see
`data/golden/README.md`). This is a hard labeling problem, not just a hard classification problem
— several of AmazonHelp's intents (`return_or_replacement_request` vs `delivery_not_received` vs
`general_other`, for example) genuinely overlap in real customer language.

## Finding 3: the 99% auto-precision claim does not survive contact with human labels

The automation-precision/coverage sweep (`scripts/evaluate_automation.py`) found a threshold
(`evidence_score >= 0.30`, `intent_confidence >= 0.80`) achieving **99.0% precision at 20.1%
coverage** — but that was measured against the same silver test-set labels as Finding 1.

Applying that *exact, untouched* threshold to the golden set instead
(`reports/golden_automation_check.json`):

| | Coverage | Auto-precision | N auto-handled |
|---|---|---|---|
| Silver-based estimate | 20.1% | **99.0%** | 502 / 2,500 |
| Golden-based measurement | 16.0% | **68.75%** (95% CI: 51.4%–82.1%) | 22 / 32 |

Even the *upper bound* of the 95% confidence interval (82.1%) is well below the claimed 99%. This
threshold was **not** re-tuned on the golden set (that would be the exact golden-set-overfitting
the assignment warns against) — it was evaluated as-is, honestly, and it does not hold up.

**Practical read:** the true auto-handling precision of this system, at this operating point, is
probably somewhere in the 50-80% range, not 99% — and 32 auto-handled golden examples is too few
to pin that down more precisely. A real deployment would need a substantially larger hand-labeled
sample before trusting a specific number here.

## Finding 4: this doesn't yet include a live LLM grounding gate

Every number above is intent-classification-only; none of it includes the `grounding_score` gate
in the escalation policy (`backend/app/escalation/policy.py`), because that requires a live LLM
call this sandbox cannot make (see README "LLM execution status"). Once a real
`LLM_PROVIDER=groq`/`gemini` is configured, the grounding check can only turn *more* AUTO cases
into ESCALATE (never the reverse) — so real-world auto-precision, once that gate is active, should
be **higher** than the intent-only numbers above, not lower. Whether it's high enough to matter is
untested here.

## Other things a single headline number hides

- **Class imbalance**: `general_other` is 28% of training data by construction (an intentional
  catch-all/abstention bucket, not a real "topic") — macro-F1, not accuracy, is the fairer summary
  statistic, and even that's reported at 0.486 on the golden set, not the misleadingly-high 0.870.
- **Rare-intent performance**: `payment_or_billing_issue` has only 245 training examples and the
  worst recall of any intent on the silver test set (0.565) — see `reports/baseline_results.md`.
- **Temporal effects**: results reflect Oct-Dec 2017 AmazonHelp behavior specifically; agent
  wording, policies, and even the intent mix could look different in a different period.
- **Retrieval quality ceiling**: Recall@1 is only 42.6% (TF-IDF has no semantic matching — see
  `reports/retrieval_metrics.json`) — evidence quality is a real bottleneck on generation
  grounding, independent of intent classification accuracy.
- **LLM judge validation**: not yet performed with a real judge — see README "LLM execution
  status." A judge's reliability cannot be assumed; it needs the same kind of check this document
  just did for the classifier.

## The honest summary

If you need one number: **~52% real-world intent accuracy (human-verified), with auto-handling
precision that has not been shown to exceed ~80% at any meaningfully large sample size** — not
88.7% accuracy, not 99% auto-precision. The system's actual strength is the *infrastructure*
(leakage-safe evaluation, evidence-first escalation, an honest golden-set check that caught its
own headline number being wrong) more than the current classifier's raw accuracy, which has clear,
measured room to improve — start there, not with a better prompt.
