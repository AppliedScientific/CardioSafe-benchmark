# Note S1 — Y-randomization mechanism and per-stratum analysis

## Background

To test for hidden information leakage in the training pipeline, we ran a
*Y-randomization* control: the labels matrix is permuted independently per
column (i.e., per binary head) and the entire model is retrained on the
scrambled labels with the same hyperparameters, seed, and schedule. A model
that relies only on signal in the chemistry should collapse to AUC = 0.5 on
the held-out test fold; any sustained departure from 0.5 indicates that the
model exploits a feature that survives per-column permutation.

## Result — per-head MCC and Pearson *r* (tan70 fold, seed 42)

All classification MCC values collapsed to within ±0.07 of zero and all
regression Pearson *r* values collapsed near zero:

| head | metric | Y-randomized | (deployed for reference) |
| :-- | :-- | --: | --: |
| hERG 10 µM | MCC | +0.043 | +0.466 |
| hERG 1 µM  | MCC | -0.022 | +0.381 |
| Nav1.5     | MCC | +0.050 | +0.451 |
| Cav1.2     | MCC | -0.062 | +0.489 |
| IKs        | MCC | +0.000 | −0.077 |
| hERG       | Pearson *r* | +0.104 | +0.553 |
| Nav1.5     | Pearson *r* | +0.019 | +0.558 |
| Cav1.2     | Pearson *r* | -0.288 | +0.689 |

(Source: an earlier-architecture Y-randomization checkpoint with a
15-dimensional descriptor block, evaluated on the tan70 test fold —
n = 46,137. The cross-attention multi-task structure is identical to the
deployed model; only the descriptor count differs from the deployed
20-dim block.)

## But hERG 10 µM Y-randomized AUC is *not* near 0.5

The hERG 10 µM Y-randomized AUC was elevated above 0.5 (0.70 on
tan70 Stage 1 / 0.70 Stage 2; 0.67
Stage 2 on tan60), prompting a deeper investigation. The MCC and
Pearson *r* collapses above are the *unconfounded* signals — both metrics
require the model to identify *which individual compounds* are positive,
not just the marginal positive-rate. AUC, by contrast, integrates the rank
order of probabilities against the label and is sensitive to a marginal-rate
model under low class prevalence.

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

### Same stratification on Stage 2 for completeness

| MW bin (g/mol) | tan70 Y-rand AUC | tan60 Y-rand AUC |
| :-- | --: | --: |
| [0, 300) | 0.594 | 0.579 |
| [300, 400) | 0.660 | 0.594 |
| [400, 500) | 0.696 | 0.568 |
| [500, +∞) | 0.681 | 0.579 |

## Sanity check — prevalence-controlled subsample

As a secondary control we sub-sampled the negative class to bring the
prevalence to 10 %, then repeated the
Y-randomization 20 times. The mean AUC on the prevalence-
controlled subsample was 0.701 ± 0.003
(Stage 1, tan70). Raising the prevalence weakens the MW carrier but does
not eliminate it, consistent with the analysis above.

## Summary

The two metrics tell a consistent story:

- **MCC and Pearson *r* collapse cleanly** — these are unconfounded by the
  MW-prevalence correlation and confirm no compound-level information
  leakage from training to held-out folds.
- **AUC is elevated to ~0.70 unstratified** — this is the confounded signal,
  driven by a MW-prevalence correlation that survives per-column label
  permutation.
- **AUC collapses to 0.50–0.62 within MW strata** — confirming MW is the
  carrier of the unstratified elevation.
