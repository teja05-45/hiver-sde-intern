# Reproducibility

## Environments

One rule is enforced: **the training environment, the evaluation environment,
and the runtime environment must be the same**, because `models/*.joblib`
embeds pickled scikit-learn estimators that are only guaranteed to load with
the scikit-learn version that produced them.

| Component | Version |
|---|---|
| Python | 3.11.9 |
| scikit-learn | 1.9.1 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.5 |
| joblib | 1.6.0 |

`requirements.txt` pins these exact versions. The backend Docker image
installs from the same file, so a Docker deployment trains/loads artifacts
with the same library stack.

## Data pipeline

`data/raw/twcs.csv` (Kaggle "Customer Support on Twitter",
`thoughtvector/customer-support-on-twitter`) is processed with fixed seeds
(`RANDOM_SEED=42`, `DATA_SAMPLE_SIZE=60000`). Every sampling step uses a
seeded RNG (see `backend/app/services/conversation.py::sample_engagement_ids`),
so re-running the pipeline reproduces the same conversations, splits, and
models. The temporal split cutoffs are fixed dates, not random fractions
(see `docs/decision-log.md` #2).

## Golden set

The 200-example golden set is a committed artifact (`data/golden/`): sampling
was stratified and seeded, corrections were applied by a documented manual
review pass (`scripts/apply_golden_corrections.py`, 94/200 labels corrected —
see `data/golden/README.md`). It is never regenerated randomly; treat it as
versioned evaluation data.

## Recomputing every reported number

Each number in `reports/` is produced by one committed script; nothing is
hand-typed. Run them in order (see README "Quickstart") to regenerate all
metrics deterministically.
