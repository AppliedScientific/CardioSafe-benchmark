# Table S2 — Comparator panel (pre-de-leak), tan70

Expanded version of main-text Table 3 with the hERG 1 µM head added.
Paired-bootstrap differences (B = 1000 resamples, common indices
applied to both models, random seed 20260428).

## AUC

| channel | comparator | n | CardioSafe | Comparator | Δ-AUC [95% CI] |
| :-- | :-- | --: | --: | --: | :-- |
| hERG 10 µM | CToxPred2 | 46,120 | 0.917 | 0.832 | +0.085 [+0.070, +0.100] |
| hERG 10 µM | CardioGenAI | 46,024 | 0.917 | 0.862 | +0.055 [+0.040, +0.070] |
| hERG 1 µM | CToxPred2 | 46,042 | 0.959 | 0.875 | +0.083 [+0.061, +0.107] |
| hERG 1 µM | CardioGenAI | 45,949 | 0.959 | 0.891 | +0.067 [+0.043, +0.090] |
| Nav1.5 | CToxPred2 | 218 | 0.831 | 0.784 | +0.047 [-0.014, +0.107] |
| Nav1.5 | CardioGenAI | 217 | 0.832 | 0.775 | +0.057 [-0.013, +0.126] |
| Cav1.2 | CToxPred2 | 160 | 0.841 | 0.815 | +0.026 [-0.065, +0.111] |
| Cav1.2 | CardioGenAI | 160 | 0.841 | 0.873 | -0.032 [-0.105, +0.039] |

## MCC

| channel | comparator | CardioSafe | Comparator | Δ-MCC [95% CI] |
| :-- | :-- | --: | --: | :-- |
| hERG 10 µM | CToxPred2 | +0.466 | +0.123 | +0.342 [+0.322, +0.363] |
| hERG 10 µM | CardioGenAI | +0.467 | +0.147 | +0.320 [+0.299, +0.342] |
| hERG 1 µM | CToxPred2 | +0.381 | +0.062 | +0.318 [+0.275, +0.363] |
| hERG 1 µM | CardioGenAI | +0.382 | +0.073 | +0.309 [+0.261, +0.351] |
| Nav1.5 | CToxPred2 | +0.451 | +0.384 | +0.068 [-0.059, +0.198] |
| Nav1.5 | CardioGenAI | +0.458 | +0.376 | +0.082 [-0.054, +0.211] |
| Cav1.2 | CToxPred2 | +0.488 | +0.471 | +0.017 [-0.153, +0.181] |
| Cav1.2 | CardioGenAI | +0.488 | +0.516 | -0.027 [-0.189, +0.119] |
