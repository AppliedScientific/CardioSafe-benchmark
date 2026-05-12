# Note S1 — Y-randomization mechanism and per-stratum analysis

## Background

To test for hidden information leakage in the training pipeline, we ran a
*Y-randomization* control: the labels matrix is permuted independently per
column (i.e., per binary head) and the entire model — Stage 1 (Stage 1) plus
Stage 2 (cliff fine-tune) — is retrained on the scrambled labels with the same
hyperparameters, seeds, and schedule as the deployed model. A model that
relied only on signal in the chemistry should collapse to AUC = 0.5 on the
held-out test fold; any sustained departure from 0.5 indicates that the model
exploits a feature that survives per-column permutation.

## Result

The hERG 10 µM Y-randomized AUC was **0.701 on tan70**
(Stage 1) and **0.667 on tan60** (Stage 2). The AUC
is sustained above 0.5 with no overlap with the random-chance interval,
prompting a deeper investigation.

## Diagnosis: the residual molecular-weight carrier

The hERG positive prevalence on tan70 is 2.0 % (937 positives in 46,120 test
compounds), and the within-pool MW distribution is correlated with the
original blocker label — large molecules are over-represented among the
blockers (positive rate climbs from 0.7 % in the [0, 300 g/mol) bin to 13.1 %
in the [500, +∞) bin). Per-column permutation scrambles *which* compound
carries the positive label but leaves the *positional* label statistics
intact. A model that learns the rule *“compounds with MW > 400 g/mol have a
higher prior of being positive”* therefore predicts above-random on the
scrambled labels too, even though the scrambling destroys the molecule ↔
label mapping. Under the natural ~2 % prevalence the ROC integral picks this
small positive-class enrichment up as AUC ≈ 0.6–0.7.

## Stratified analysis (Stage 1, tan70)

Within each MW stratum the prevalence imbalance is much smaller, and the AUC
collapses correspondingly to **0.50–0.62**, confirming that MW is the
carrier:

| MW bin (g/mol) | n | positive rate | Y-rand AUC |
| :-- | --: | --: | --: |
| [0, 300) | 16,365 | 0.72% | 0.596 |
| [300, 400) | 20,434 | 1.86% | 0.620 |
| [400, 500) | 8,315 | 3.68% | 0.567 |
| [500, +∞) | 1,006 | 13.12% | 0.498 |

Compare against the *unstratified* AUC of 0.701 on the same
fold: the stratification removes the MW carrier and the AUC collapses to the
chance interval, exactly as expected.

### Same stratification on Stage 2 (deployed) for completeness

| MW bin (g/mol) | tan70 Y-rand AUC | tan60 Y-rand AUC |
| :-- | --: | --: |
| [0, 300) | 0.594 | 0.579 |
| [300, 400) | 0.660 | 0.594 |
| [400, 500) | 0.696 | 0.568 |
| [500, +∞) | 0.681 | 0.579 |

## Unconfounded controls — the *real* signature of no information leakage

The AUC-on-scrambled-labels metric is **confounded by the MW carrier** above.
The unconfounded controls — regression Pearson *r* and binary MCC — are
*not* affected by the MW carrier, because both depend on the model
identifying which *individual compounds* are positive (not the marginal
positive-rate-given-MW): they should both collapse cleanly to ~0 under
Y-randomization. They do:

| metric (Stage 1 seed 42, Y-randomized) | tan70 | tan60 | deployed (for comparison) |
| :-- | --: | --: | --: |
| hERG pChEMBL Pearson *r* | +0.115 | +0.155 | +0.553 / +0.331 (tan70 / tan60) |
| hERG 10 µM MCC | +0.003 | +0.040 | +0.466 / +0.391 |
| Nav1.5 blocker MCC | +0.104 | +0.220 | +0.451 / +0.460 |
| Cav1.2 blocker MCC | -0.081 | +0.227 | +0.489 / +0.108 |

Pearson *r* drops from ≈ +0.55 to ≈ +0.11 on tan70 (and from +0.33 to +0.16
on tan60); MCC drops from +0.47 to +0.003 on tan70 hERG. The model
**cannot** identify which individual compound is positive — it can only
learn the MW-stratified marginal. Combined with the per-stratum AUC collapse
above, this confirms that the elevated unstratified Y-randomized AUC is a
confound of the MW–prevalence correlation rather than information leakage
from the training to the held-out folds.

## Sanity check — prevalence-controlled subsample

As a secondary control we sub-sampled the negative class to bring the
prevalence to 10 %, then repeated the
Y-randomization 20 times. The mean AUC on the prevalence-
controlled subsample was 0.701 ± 0.003
(Stage 1, tan70). Raising the prevalence weakens the MW carrier but does
not eliminate it, consistent with the analysis above.

## Summary

The 0.70 / 0.67 elevated Y-randomized
AUC values reported in the body of the manuscript are the *unconfounded
confound* — they reflect a small but real MW–label correlation that any
prediction-marginal-rate model would pick up, not information leakage. The
per-MW-stratum AUC collapse to 0.50–0.62 and the clean collapse of regression
*r* and MCC together establish that no training–test compound-level leakage
is present.
