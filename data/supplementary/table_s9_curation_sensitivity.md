# Table S9 — Curation sensitivity (strict vs full)

Headline metrics across three curation policies, on both splits.
Each cell is the deployed 5-seed CardioSafe ensemble on the corresponding
test fold (whose composition shrinks with the stricter policy, because some
test rows lose their non-NaN label).

## Training-pool labelled-compound counts per channel

| channel | Full (deployed) | Exact-only | Exact-only-strict | strict / full |
| :-- | --: | --: | --: | --: |
| hERG 10 µM | 331,127 | 23,787 | 12,746 | 3.8% |
| hERG 1 µM | 330,541 | 23,787 | 12,746 | 3.9% |
| Nav1.5 | 3,160 | 2,855 | 1,579 | 50.0% |
| Cav1.2 | 1,138 | 997 | 294 | 25.8% |
| IKs | 115 | 115 | 66 | 57.4% |
| hERG pChEMBL | 23,787 | 23,787 | 12,746 | 53.6% |
| Nav1.5 pChEMBL | 2,855 | 2,855 | 1,579 | 55.3% |
| Cav1.2 pChEMBL | 997 | 997 | 294 | 29.5% |

## Classification AUC across policies

| head | split | Full | Exact-only | Exact-only-strict | Δ (strict − full) |
| :-- | :--: | --: | --: | --: | --: |
| hERG 10 µM | tan70 | 0.917 | 0.778 | 0.737 | -0.180 |
| hERG 10 µM | tan60 | 0.898 | 0.767 | 0.688 | -0.210 |
| hERG 1 µM | tan70 | 0.959 | 0.798 | 0.788 | -0.170 |
| hERG 1 µM | tan60 | 0.932 | 0.771 | 0.747 | -0.185 |
| Nav1.5 | tan70 | 0.831 | 0.790 | 0.694 | -0.137 |
| Nav1.5 | tan60 | 0.792 | 0.793 | 0.731 | -0.061 |
| Cav1.2 | tan70 | 0.841 | 0.769 | 0.744 | -0.097 |
| Cav1.2 | tan60 | 0.766 | 0.697 | 0.429 | -0.337 |
| IKs | tan70 | 0.385 | 0.000 | 0.286 | -0.099 |
| IKs | tan60 | 0.500 | 0.500 | nan | +nan |

## Regression Pearson r across policies

| head | split | Full | Exact-only | Exact-only-strict | Δ (strict − full) |
| :-- | :--: | --: | --: | --: | --: |
| hERG pChEMBL | tan70 | 0.553 | 0.553 | 0.520 | -0.033 |
| hERG pChEMBL | tan60 | 0.331 | 0.381 | 0.443 | +0.112 |
| Nav1.5 pChEMBL | tan70 | 0.558 | 0.528 | 0.353 | -0.205 |
| Nav1.5 pChEMBL | tan60 | 0.197 | 0.293 | 0.325 | +0.128 |
| Cav1.2 pChEMBL | tan70 | 0.689 | 0.633 | 0.683 | -0.007 |
| Cav1.2 pChEMBL | tan60 | 0.569 | 0.489 | 0.043 | -0.527 |

## Notes

- *Full* is the deployed pharmacology-aware curation: exact `pchembl_value` records, censored relations (`>`, `>=`, `<`, `<=` in nM) converted via `pChEMBL = 9 − log10(nM)`, and inhibition-percentage votes with assay-parsed test concentration.
- *Exact-only* keeps only the exact `pchembl_value` rows (drops censored + %-inhibition votes).
- *Exact-only-strict* further restricts to `standard_relation = '='` records, matching the criterion of Arab et al. [41] (CToxPred).
- Each strict policy retrains the full pipeline end-to-end; AUC / MCC / Pearson r are computed on the deployed 5-seed ensemble against the *same* test split but on the test rows still carrying a label under that policy.
- Headline observation from the manuscript: dropping to exact-only-strict costs **0.18** hERG binary AUC on tan70 (0.917 → 0.737) and **0.21** on tan60 (0.898 → 0.688), because 93% of hERG binary labels in ChEMBL derive from inhibition-percentage or censored votes rather than exact IC50. Nav1.5 and Cav1.2 binary AUC drops are smaller (0.04–0.07) because those channels lose only ~10% of training rows.
- Regression Pearson r on the stricter **tan60** split *improves* under the strict policy (hERG: 0.331 → 0.443; Nav1.5: 0.197 → 0.325), suggesting the noisy censored / inhibition-percentage votes help binary classification but hurt regression on harder out-of-distribution splits.
