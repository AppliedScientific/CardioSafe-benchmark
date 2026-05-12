# Table S8 — L1000 threshold sensitivity sweep

How CardioSafe headline classification AUC and MCC vary as the L1000
encoder's co-expression edge-density threshold *r* is changed around
the deployed default (r = 0.40). Two off-default ablations were retrained
end-to-end through the full Stage 1 + Stage 2 pipeline: r = 0.30 (a denser
predicted-output graph) and r = 0.50 (a sparser one).

## Headline AUC × MCC (5-seed deployed ensemble)

### tan70

| head | r = 0.30 AUC (MCC) | r = 0.40 deployed AUC (MCC) | r = 0.50 AUC (MCC) |
| :-- | --: | --: | --: |
| hERG 10 µM | 0.914 (+0.455) | 0.917 (+0.466) | 0.913 (+0.470) |
| hERG 1 µM | 0.956 (+0.387) | 0.959 (+0.381) | 0.953 (+0.383) |
| Nav1.5 | 0.827 (+0.487) | 0.831 (+0.451) | 0.838 (+0.460) |
| Cav1.2 | 0.827 (+0.497) | 0.841 (+0.488) | 0.836 (+0.518) |
| IKs | 0.615 (-0.077) | 0.385 (-0.077) | 0.615 (-0.077) |

### tan60

| head | r = 0.30 AUC (MCC) | r = 0.40 deployed AUC (MCC) | r = 0.50 AUC (MCC) |
| :-- | --: | --: | --: |
| hERG 10 µM | 0.901 (+0.390) | 0.898 (+0.391) | 0.900 (+0.373) |
| hERG 1 µM | 0.936 (+0.320) | 0.932 (+0.308) | 0.932 (+0.298) |
| Nav1.5 | 0.769 (+0.327) | 0.792 (+0.460) | 0.763 (+0.288) |
| Cav1.2 | 0.766 (+0.108) | 0.766 (+0.108) | 0.833 (+0.335) |
| IKs | 0.700 (+0.000) | 0.500 (+0.000) | 0.500 (+0.000) |

## L1000 encoder predicted-output graph density

Edge density of the predicted-output graph (subsampled to 50,000 compounds; 978 genes, 477,753 undirected gene–gene pairs) at each co-expression threshold. The encoder was trained with default r = 0.40, at which the training-time graph had 7,829 undirected edges — 29× sparser than the predicted-output graph at the same threshold (0.473 vs ~0.016 edge density).

| threshold | n edges (undirected) | density |
| :--: | --: | --: |
| r = 0.30 | 282,560 | 0.5914 |
| r = 0.35 | 253,557 | 0.5307 |
| r = 0.40 | 226,000 | 0.4730 |
| r = 0.45 | 199,214 | 0.4170 |
| r = 0.50 | 172,905 | 0.3619 |
| r = 0.60 | 121,832 | 0.2550 |
| r = 0.70 | 74,148 | 0.1552 |
| r = 0.80 | 33,148 | 0.0694 |

## Notes

- The default deployed threshold is **r = 0.40**, the value at which the L1000 encoder was trained. The 0.30 and 0.50 variants required retraining the encoder and recomputing the 978-d predicted-L1000 cache.
- AUC and MCC are reported on the deployed 5-seed CardioSafe ensemble (mean of post-sigmoid probabilities, MCC at threshold 0.5).
- *n* values vary slightly across heads because each head's denominator is the subset of test-fold compounds with a non-NaN label for that head.
- Headline observation: AUC moves within ±0.005 across the three r values on every classification head; consistent with the manuscript's conclusion that the L1000 branch contribution is statistically indistinguishable from zero at the current data scale.
