# Table S5 — 11-drug panel on tan60

Same 11 drugs and same schema as main-text Table 5, but evaluated on the
tan60 split (5-seed deployed CardioSafe ensemble). CO = classifier output
(post-sigmoid probability, ensemble mean across the 5 seeds).

Under the tan60 cutoff, Astemizole, Sertindole, Verapamil, Amiodarone, and
Ranolazine are in the training fold (max Tanimoto = 1.000 to a training
compound); the remaining six drugs are out-of-train.

## Headline summary

| Drug | Class | In train? | max Tan | hERG pIC50 | hERG 10 µM CO | Nav1.5 CO | Cav1.2 CO |
| :-- | :-- | :--: | --: | --: | --: | --: | --: |
| Terfenadine | withdrawn | no | 0.891 | 6.05 | 0.887 | 0.556 | 0.498 |
| Fexofenadine | safe metabolite | no | 0.820 | 4.63 | 0.745 | 0.381 | 0.376 |
| Cisapride | withdrawn | no | 0.627 | 7.20 | 0.917 | 0.505 | 0.388 |
| Astemizole | withdrawn | yes | 1.000 | 8.40 | 0.983 | 0.695 | 0.430 |
| Sertindole | withdrawn | yes | 1.000 | 6.21 | 0.912 | 0.478 | 0.444 |
| Grepafloxacin | withdrawn | no | 0.590 | 3.85 | 0.590 | 0.193 | 0.103 |
| Verapamil | safe multichannel | yes | 1.000 | 6.65 | 0.749 | 0.475 | 0.579 |
| Amiodarone | safe multichannel | yes | 1.000 | 5.03 | 0.760 | 0.556 | 0.540 |
| Ranolazine | safe multichannel | yes | 1.000 | 4.96 | 0.719 | 0.404 | 0.303 |
| Ondansetron | black-box QT | no | 0.250 | 4.14 | 0.413 | 0.187 | 0.082 |
| Domperidone | black-box QT | no | 0.885 | 6.24 | 0.887 | 0.497 | 0.361 |

## Full pIC50 + CO with per-seed standard deviations

| Drug | hERG pIC50 (SD) | Nav1.5 pIC50 (SD) | Cav1.2 pIC50 (SD) | hERG 10 µM CO (SD) | hERG 1 µM CO (SD) | Nav1.5 CO (SD) | Cav1.2 CO (SD) | IKs CO (SD) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| Terfenadine | 6.05 (0.21) | 5.09 (0.03) | 5.18 (0.09) | 0.887 (0.005) | 0.756 (0.039) | 0.556 (0.023) | 0.498 (0.071) | 0.319 (0.076) |
| Fexofenadine | 4.63 (0.14) | 4.78 (0.09) | 4.92 (0.18) | 0.745 (0.025) | 0.528 (0.031) | 0.381 (0.045) | 0.376 (0.059) | 0.307 (0.034) |
| Cisapride | 7.20 (0.18) | 5.05 (0.05) | 4.97 (0.23) | 0.917 (0.021) | 0.858 (0.038) | 0.505 (0.050) | 0.388 (0.085) | 0.229 (0.082) |
| Astemizole | 8.40 (0.06) | 5.30 (0.18) | 4.93 (0.39) | 0.983 (0.012) | 0.970 (0.008) | 0.695 (0.098) | 0.430 (0.195) | 0.144 (0.065) |
| Sertindole | 6.21 (0.37) | 5.00 (0.13) | 5.08 (0.10) | 0.912 (0.041) | 0.795 (0.050) | 0.478 (0.071) | 0.444 (0.090) | 0.242 (0.029) |
| Grepafloxacin | 3.85 (0.14) | 4.48 (0.22) | 4.35 (0.29) | 0.590 (0.083) | 0.353 (0.096) | 0.193 (0.122) | 0.103 (0.049) | 0.219 (0.115) |
| Verapamil | 6.65 (0.08) | 4.92 (0.06) | 5.71 (0.40) | 0.749 (0.098) | 0.636 (0.112) | 0.475 (0.032) | 0.579 (0.101) | 0.299 (0.084) |
| Amiodarone | 5.03 (0.05) | 5.15 (0.11) | 5.59 (0.37) | 0.760 (0.025) | 0.480 (0.052) | 0.556 (0.054) | 0.540 (0.150) | 0.321 (0.041) |
| Ranolazine | 4.96 (0.07) | 4.84 (0.04) | 4.80 (0.16) | 0.719 (0.029) | 0.455 (0.022) | 0.404 (0.050) | 0.303 (0.058) | 0.318 (0.068) |
| Ondansetron | 4.14 (0.12) | 4.46 (0.17) | 4.46 (0.34) | 0.413 (0.055) | 0.168 (0.053) | 0.187 (0.134) | 0.082 (0.050) | 0.225 (0.103) |
| Domperidone | 6.24 (0.28) | 5.04 (0.04) | 5.01 (0.11) | 0.887 (0.030) | 0.772 (0.039) | 0.497 (0.038) | 0.361 (0.086) | 0.325 (0.049) |

## Notes

- *pIC50* is the 5-seed-mean regression output of the deployed CardioSafe ensemble; the value in parentheses is the per-seed standard deviation.
- *CO* (classifier output) is the 5-seed-mean sigmoid output of each binary head.
- *In train?* means: at least one training-fold compound has Morgan-r=2 2048-bit Tanimoto = 1.000 to this drug under the tan60 cutoff (i.e., InChI-key identical or near-identical).
- *max Tan* is the maximum Tanimoto similarity from this drug to any compound in the tan60 training fold.
- Compare directly with main-text Table 5 (same drugs on tan70). Notably, Ranolazine is *out-of-train on tan70* (max Tan 0.588) and *in-train on tan60* (max Tan 1.000); its hERG 10 µM CO moves from 0.123 on tan70 to 0.719 on tan60 — a within-distribution-vs-out-of-distribution illustration.
