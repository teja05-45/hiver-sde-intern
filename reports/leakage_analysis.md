# Leakage Analysis

Brand: AmazonHelp
Total labeled conversations: 44375
Split sizes: train=28228, dev=8587, test=7560
Split cutoffs: train < 2017-11-10, dev in [2017-11-10, 2017-11-25), test >= 2017-11-25

## Duplicate analysis (normalized-text exact matches)
- Duplicate rate: 44.92% of documents belong to a duplicate group
- 8186 duplicate groups found (19935 documents total)

## Cross-split leakage comparison

| Split strategy | Duplicate groups spanning splits | Documents involved |
|---|---|---|
| Temporal (this project) | 16 | 65 |
| Naive random (for comparison) | 2701 | 7365 |
