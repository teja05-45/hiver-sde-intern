# Golden-Set Gap Analysis — why golden accuracy is 19.5% while silver shows 93.6%

*Written 2026-09-15. Every number below was measured in this environment from the
committed artifacts; the commands to reproduce are listed at the end.*

## 1. Observed gap

| Metric (current committed artifacts) | Value |
|---|---|
| Silver test accuracy (`reports/baseline_results.json`, regenerated 2026-09-14) | **93.57%** |
| Golden accuracy (`reports/golden_set_evaluation.json`, regenerated 2026-09-14) | **19.50%** (39/200) |
| Reported "gap" (`silver_vs_golden_comparison.gap`) | 0.7407 |
| Golden accuracy claimed in README/final_report before 2026-09-14 | 53.5% (107/200) |
| Failure count claimed before 2026-09-14 vs. current `failure_analysis.json` | 93 vs. **161** |

This is not a classifier regression in isolation; it is a **version mismatch between
artifacts** produced at different times from two different label regimes.

## 2. Potential causes investigated (and eliminated)

| Hypothesis | Test run | Result |
|---|---|---|
| Label-mapping/taxonomy bug in evaluation join | Rejoined `golden_set.csv` against `golden_set_with_predictions.json` | 200/200 messages and labels identical; join is clean |
| Wrong/stale classifier artifact | Loaded committed joblib AND retrained fresh from current `labeled_AmazonHelp.jsonl` | **Identical predictions; both 19.50%** |
| Golden CSV clobbered by the 2026-09-12 overwrite incident | Checked provenance counts (86/107/7) and per-row cross-check vs the verified predictions export | Golden CSV is the verified, human-reviewed state — labels did NOT change |
| Preprocessing mismatch (train vs golden) | `clean_for_modeling` applied identically in both paths | Not a factor |
| Label-mapping bug in the evaluation script itself | Re-computed accuracy manually from per-example predictions | Matches 39/200 exactly |
| Human labels are wrong | Spot-read the corrected labels against `configs/intents.yaml` definitions | Labels are coherent (the classifier predicts `delivery_*` for complaint/refund/OOD messages exactly as a TF-IDF classifier trained on remapped labels would) |

## 3. Root cause (isolated, with evidence)

**The silver labels changed under the golden evaluation; the narrative was never updated.**

Timeline reconstruction from file mtimes and in-file provenance:

1. **2026-09-12 ~22:39** — the verified golden-set state existed: 53.5% golden accuracy,
   12 intents everywhere, failure analysis with 93 failures. All documents written that
   day agree (`README.md`, `reports/final_report.md`, `data/golden/README.md`,
   `FINAL_VERIFICATION.md`).
2. **2026-09-12 23:32** — `configs/intents.yaml` edited: the cluster→intent mapping was
   re-derived for a re-run of clustering with k=15 (the YAML header says "RE-MAPPED
   2026-09-12 … see docs/decision-log.md #19" — a decision entry that was never written).
3. **2026-09-12 23:33** — `scripts/build_labeled_dataset.py` re-run, rewriting the silver
   labels of the whole dataset (same conversations, same cluster-assignment file from
   21:03; only the *mapping* changed).
4. **2026-09-14 22:27** — `scripts/train_and_save_agent.py` re-run: the committed
   classifier/retriever artifacts are trained on the REMAPPED silver labels.
5. **2026-09-14 23:15–23:19** — `evaluate_baselines.py`, `evaluate_against_golden.py`,
   `analyze_failures.py` re-run → `baseline_results.json` (93.57% silver, 11 test intents),
   `golden_set_evaluation.json` (19.5% golden), `failure_analysis.json` (161 failures).

Measured decomposition (all in this environment):

- Current silver labels vs. human golden labels **on the same 200 messages**:
  agreement **43/200 (21.5%)**. The golden labels did not move — the silver labels did.
- By golden label provenance: silver agrees with **manual-review** labels 10/86 (11.6%),
  with classifier-fallback labels 27/107 (25.2%), with keyword-rule labels 6/7 (85.7%).
- When silver disagrees with the human label, the most frequent silver labels are
  `delivery_not_received` (59) and `delivery_delay` (32) — consistent with the remapped
  split of cluster 4 ("delivered-but-missing vs wrong-item") and with the new rule that
  sends **all of cluster 14 (52.4% of the corpus)** to `general_other`.
- Current test split: 47.8% of messages carry `general_other`; `account_access_issue`
  has **6 training examples** (the old reports show it with 7 golden + full taxonomy
  support; the current `baseline_results.json` has no test support for it at all).

Why the classifier scores 19.5%: it is trained on labels that (a) merged most real
traffic into `general_other`/delivery buckets and (b) re-split cluster 4 differently
than the human labeler would. It cannot learn human label boundaries from silver labels
that disagree with them 78.5% of the time. Silver accuracy stays 93.6% because silver is
graded by the same remapped process — the exact circularity this project documented in
`reports/misleading_headline_number.md`, now one level deeper.

Why the OLD 53.5% also deserves a caveat: the golden labels themselves are 53.5%
classifier-fallback-derived (107/200) — the golden set's draft labels came from the
pre-remap classifier, corrected for 86/200 rows by manual review. The 53.5% therefore
**overstates** true accuracy (the classifier is graded partly against its own old
predictions); the honest independent estimate from that era is the manual+keyword
subset. This caveat was never surfaced before; it is documented here and in
`data/golden/README.md`.

## 4. Fix applied

No code path could recover the pre-remap `intents.yaml` (no git history in this
environment; the old file is gone). Options considered:

- **(chosen) Make the whole artifact chain consistent with the current dataset and
  document the regression honestly.** Regenerate every report from the current
  dataset/model, fix the README/final-report numbers, mark the 53.5% era as
  invalidated by the artifact mismatch, and record the remap as decision #19 with its
  measured cost. This preserves the project's core value: truthful evaluation.
- (rejected) Re-invent a plausible old mapping to "restore 53.5%" — fabricating an
  unverified mapping would violate the project's no-fabrication rule.
- (rejected) Hide the gap cosmetically — prohibited by the assignment.

Concretely: all documents now quote **19.5% golden / 93.6% silver / 74.1-point gap**
from the current consistent artifacts, describe the 53.5% state as the pre-remap
measurement invalidated by the Sep-12 dataset rebuild, and state the per-source
circularity caveat. The remap itself is recorded as decision #19 in
`docs/decision-log.md` with its measured consequences.

## 5. Before/after (artifact consistency)

| Artifact | Before fix (2026-09-15 audit) | After fix |
|---|---|---|
| `README.md` §5/§6 | 53.5% golden / 40.3-pt gap / 93.8% silver | current numbers (19.5% / 74.1-pt / 93.6%) + artifact-mismatch note |
| `reports/final_report.md` | 53.5% / 40.3-pt / 93 failures | current numbers / 161 failures + root-cause link |
| `data/golden/README.md` | "86/200 corrected" claim without circularity caveat | caveat added (53.5% of golden draft labels were classifier-derived) |
| `docs/decision-log.md` | 18 decisions; "#19" referenced but missing | 19 decisions, #19 documents the remap and its measured cost |
| `reports/golden_set_evaluation.json` / `baseline_results.json` / `failure_analysis.json` | internally consistent (19.5%) | unchanged data, now consistently described everywhere |

## 6. Reproduce

```bash
venv/Scripts/python.exe - <<'PY'
import json, csv, joblib, sys
sys.path.insert(0, "backend")
from app.services.text_cleaning import clean_for_modeling
clf = joblib.load("models/classifier_AmazonHelp.joblib")
golden = list(csv.DictReader(open("data/golden/golden_set.csv", encoding="utf-8")))
preds = [p.intent for p in clf.predict([clean_for_modeling(r["message"]) for r in golden])]
print(sum(p == r["intent"] for p, r in zip(preds, golden)) / len(golden))  # 0.195
PY
```

ROOT CAUSE: ISOLATED — silver-label remapping (configs/intents.yaml, 2026-09-12 23:32)
plus a dataset rebuild one minute later invalidated every golden-set number produced
after it; the 2026-09-14 report regeneration propagated the mismatch into the reports
while the narrative documents still described the pre-remap state.
