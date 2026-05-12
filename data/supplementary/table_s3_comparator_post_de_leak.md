# Table S3 — Comparator panel (post-de-leak), tan70

Expanded version of main-text Table 3b with the hERG 1 µM head added.
Post-de-leak = test fold minus comparator-train compounds at Tanimoto
≥ 0.99. Paired-bootstrap differences (B = 1000 resamples, common
indices, random seed 20260428).

## AUC

| channel | comparator | n total | n leak | leak % | n kept | CardioSafe | Comparator | Δ-AUC [95% CI] |
| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |
| hERG 10 µM | CToxPred2 | 46,120 | 1,306 | 2.8% | 44,814 | 0.899 | 0.768 | +0.131 [+0.107, +0.155] |
| hERG 10 µM | CardioGenAI | 46,024 | 1,299 | 2.8% | 44,725 | 0.899 | 0.783 | +0.115 [+0.088, +0.143] |
| hERG 1 µM | CToxPred2 | 46,042 | 1,306 | 2.8% | 44,736 | 0.951 | 0.812 | +0.138 [+0.098, +0.178] |
| hERG 1 µM | CardioGenAI | 45,949 | 1,299 | 2.8% | 44,650 | 0.951 | 0.831 | +0.120 [+0.085, +0.157] |
| Nav1.5 | CToxPred2 | 218 | 48 | 22.0% | 170 | 0.821 | 0.761 | +0.061 [-0.002, +0.134] |
| Nav1.5 | CardioGenAI | 217 | 48 | 22.1% | 169 | 0.823 | 0.747 | +0.075 [+0.003, +0.146] |
| Cav1.2 | CToxPred2 | 160 | 33 | 20.6% | 127 | 0.877 | 0.771 | +0.106 [+0.032, +0.194] |
| Cav1.2 | CardioGenAI | 160 | 33 | 20.6% | 127 | 0.877 | 0.855 | +0.022 [-0.050, +0.100] |

## MCC

| channel | comparator | n kept | CardioSafe | Comparator | Δ-MCC [95% CI] |
| :-- | :-- | --: | --: | --: | :-- |
| hERG 10 µM | CToxPred2 | 44,814 | +0.399 | +0.072 | +0.327 [+0.298, +0.353] |
| hERG 10 µM | CardioGenAI | 44,725 | +0.401 | +0.086 | +0.314 [+0.284, +0.342] |
| hERG 1 µM | CToxPred2 | 44,736 | +0.341 | +0.037 | +0.304 [+0.244, +0.359] |
| hERG 1 µM | CardioGenAI | 44,650 | +0.343 | +0.046 | +0.297 [+0.244, +0.356] |
| Nav1.5 | CToxPred2 | 170 | +0.421 | +0.330 | +0.091 [-0.053, +0.240] |
| Nav1.5 | CardioGenAI | 169 | +0.428 | +0.315 | +0.113 [-0.033, +0.254] |
| Cav1.2 | CToxPred2 | 127 | +0.550 | +0.405 | +0.146 [-0.022, +0.321] |
| Cav1.2 | CardioGenAI | 127 | +0.550 | +0.487 | +0.063 [-0.092, +0.221] |
