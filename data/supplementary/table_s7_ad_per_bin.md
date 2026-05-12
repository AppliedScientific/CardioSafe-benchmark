# Table S7 — Applicability-domain per-bin metrics

Classification performance binned by nearest-neighbour Tanimoto similarity
to the training fold, for each binary head on tan70 and tan60. Bootstrap
95% CIs (B = 1000 resamples, seed 20260428). Bins are right-open intervals
on Morgan-r=2 2048-bit Tanimoto. The `0.7–1.0` bin is empty by construction
of the Tanimoto-cutoff splits (no test compound has a training neighbour at
or above the split's cutoff).

## tan70

| head | bin | n | prevalence | AUC [95% CI] | MCC [95% CI] |
| :-- | :--: | --: | --: | :-- | :-- |
| hERG 10 µM | 0.0–0.3 | 513 | 4.09% | 0.812 [0.696, 0.911] | +0.218 [+0.070, +0.358] |
| hERG 10 µM | 0.3–0.5 | 8,990 | 3.69% | 0.887 [0.866, 0.907] | +0.386 [+0.341, +0.426] |
| hERG 10 µM | 0.5–0.7 | 36,617 | 1.59% | 0.920 [0.904, 0.935] | +0.518 [+0.489, +0.546] |
| hERG 1 µM | 0.0–0.3 | 501 | 0.20% | 0.844 [0.812, 0.876] | -0.003 [-0.007, +0.000] |
| hERG 1 µM | 0.3–0.5 | 8,954 | 0.64% | 0.910 [0.862, 0.953] | +0.223 [+0.143, +0.307] |
| hERG 1 µM | 0.5–0.7 | 36,587 | 0.40% | 0.970 [0.949, 0.986] | +0.443 [+0.385, +0.496] |
| Nav1.5 | 0.0–0.3 | 10 | 10.00% | 0.667 [0.333, 1.000] | -0.167 [-0.486, +0.000] |
| Nav1.5 | 0.3–0.5 | 71 | 16.90% | 0.806 [0.695, 0.907] | +0.452 [+0.179, +0.674] |
| Nav1.5 | 0.5–0.7 | 137 | 35.04% | 0.828 [0.756, 0.896] | +0.443 [+0.289, +0.594] |
| Cav1.2 | 0.0–0.3 | 9 | 22.22% | 0.500 [0.000, 1.000] | +0.189 [-0.478, +0.776] |
| Cav1.2 | 0.3–0.5 | 66 | 21.21% | 0.636 [0.467, 0.783] | +0.023 [-0.200, +0.271] |
| Cav1.2 | 0.5–0.7 | 85 | 32.94% | 0.940 [0.881, 0.985] | +0.748 [+0.624, +0.864] |
| IKs | 0.5–0.7 | 7 | 14.29% | 0.333 [0.000, 0.694] | -0.167 [-0.471, +0.000] |

## tan60

| head | bin | n | prevalence | AUC [95% CI] | MCC [95% CI] |
| :-- | :--: | --: | --: | :-- | :-- |
| hERG 10 µM | 0.0–0.3 | 282 | 4.61% | 0.825 [0.691, 0.929] | +0.319 [+0.181, +0.451] |
| hERG 10 µM | 0.3–0.5 | 4,651 | 4.77% | 0.885 [0.856, 0.910] | +0.396 [+0.355, +0.437] |
| hERG 10 µM | 0.5–0.7 | 8,898 | 2.24% | 0.902 [0.873, 0.927] | +0.381 [+0.333, +0.431] |
| hERG 1 µM | 0.3–0.5 | 4,605 | 1.04% | 0.919 [0.878, 0.950] | +0.238 [+0.154, +0.320] |
| hERG 1 µM | 0.5–0.7 | 8,886 | 0.57% | 0.936 [0.893, 0.973] | +0.381 [+0.297, +0.461] |
| Nav1.5 | 0.3–0.5 | 55 | 16.36% | 0.838 [0.716, 0.934] | +0.480 [+0.147, +0.764] |
| Nav1.5 | 0.5–0.7 | 27 | 44.44% | 0.711 [0.478, 0.903] | +0.331 [-0.054, +0.682] |
| Cav1.2 | 0.3–0.5 | 37 | 16.22% | 0.780 [0.537, 0.956] | +0.041 [-0.213, +0.422] |
| Cav1.2 | 0.5–0.7 | 19 | 31.58% | 0.756 [0.500, 0.962] | +0.205 [-0.267, +0.676] |
| IKs | 0.3–0.5 | 9 | 11.11% | 0.625 [0.250, 1.000] | +0.000 [+0.000, +0.000] |

## Notes

- *Bin* is `max_Tan_to_train ∈ [lo, hi)` — nearest-neighbour Morgan-r=2 2048-bit Tanimoto from the test compound to *any* compound in the training fold for that split.
- *Prevalence* is the in-bin positive-class fraction for the binary head.
- AUC and MCC are computed on the deployed 5-seed CardioSafe ensemble (mean of post-sigmoid probabilities per head, thresholded at 0.5 for MCC).
- Bins with no compounds for a (split, head) combination are omitted from the tables; the CSV / JSON dumps retain them with null fields.
- Headline observation from the manuscript: hERG 1 µM in the `[0.0–0.3)` bin on tan70 keeps AUC at 0.844 but MCC collapses to −0.003 — the deployed classifier should not be relied upon at the default threshold below Tanimoto 0.30.
