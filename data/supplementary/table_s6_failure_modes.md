# Table S6 — Failure-mode representative SMILES

Representative compounds drawn from the top-confident false predictions
across the 5 binary heads on tan70 and tan60. Each compound is classified
into one of six failure-mode buckets per the *Failure-mode analysis*
section of the manuscript:

- **(i) Prodrug motif** (`i_prodrug`) — Hydrolyzable group; active metabolite would behave differently
- **(ii) Multi-target liability** (`ii_multi_target`) — Compound is a blocker on another cardiac channel; the label on this head is correct but the prediction confuses cross-channel signal
- **(iii) Label noise** (`iii_label_noise`) — Inconsistent or physically implausible label (e.g., 1 µM = blocker but 10 µM = non-blocker; or label disagrees with herg_pchembl by > 1 log unit)
- **(iv) Outside applicability domain** (`iv_AD`) — Max Tanimoto < 0.30 to any training compound — model should not be trusted here at the default operating point
- **(v) True in-domain model failure** (`v_true_failure`) — Compound is in-distribution (max Tan ≥ 0.50) with no liability or label-noise signal — a genuine model failure
- **Borderline** (`borderline`) — Max Tanimoto 0.30–0.50 — applicability-domain boundary, neither clearly in nor out of the trained space

## Table

| split | head | bucket | kind | true | pred prob | max Tan | SMILES | rationale |
| :--: | :-- | :-- | :--: | :--: | --: | --: | :-- | :-- |
| tan70 | hERG 10 µM | (i) Prodrug motif | FN | 1 | 0.0384 | 0.657 | `O=C(CSc1cccc[n+]1[O-])Nc1cccc(S(=O)(=O)NC2=NCCCCC2)c1` | SMARTS:schiff_base |
| tan70 | hERG 10 µM | (i) Prodrug motif | FN | 1 | 0.0500 | 0.661 | `CSCCC(NC(=O)c1ccc(Cl)cc1)C(=O)OCC(C)=O` | SMARTS:carboxylic_ester |
| tan70 | hERG 10 µM | (i) Prodrug motif | FN | 1 | 0.0534 | 0.500 | `CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12` | SMARTS:carboxylic_ester |
| tan70 | hERG 10 µM | (iii) Label noise | FP | 0 | 0.9018 | 0.635 | `OC(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1` | label=0 but herg_pchembl=5.59 (≥5.5) |
| tan70 | hERG 10 µM | (iii) Label noise | FP | 0 | 0.8885 | 0.615 | `CCNC(=O)N1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1` | label=0 but herg_pchembl=7.25 (≥5.5) |
| tan70 | hERG 10 µM | (v) True in-domain model failure | FN | 1 | 0.0285 | 0.661 | `Cc1ccc(S(=O)(=O)NCCCCCC(=O)Nc2c(C)n(C)n(-c3ccccc3)c2=O)cc1` | max_tan=0.661 ≥ 0.50, no liability/noise signal |
| tan70 | hERG 10 µM | (v) True in-domain model failure | FN | 1 | 0.0472 | 0.698 | `O=C(CC1SC(N/N=C/c2cccs2)=NC1=O)Nc1ccccc1F` | max_tan=0.698 ≥ 0.50, no liability/noise signal |
| tan70 | hERG 10 µM | (v) True in-domain model failure | FN | 1 | 0.0530 | 0.574 | `Cc1sc2ncnc(SCC(=O)NC(=O)c3cccn3C)c2c1C` | max_tan=0.574 ≥ 0.50, no liability/noise signal |
| tan70 | hERG 10 µM | Borderline | FP | 0 | 0.9237 | 0.494 | `Cc1cc(CCN2CCN(c3nsc4ccccc34)CC2)cc2c1NC(=O)CC2(C)C` | max_tan=0.494 (0.30–0.50 borderline AD) |
| tan70 | hERG 10 µM | Borderline | FN | 1 | 0.0770 | 0.383 | `Cc1ccc(S(=O)(=O)C(C#N)=C2N(C)c3ccccc3N2C)cc1` | max_tan=0.383 (0.30–0.50 borderline AD) |
| tan70 | hERG 10 µM | Borderline | FN | 1 | 0.0778 | 0.431 | `CC1=CC(=O)C(C(C)C)=CC1=NC(=O)c1ccc(C)cc1` | max_tan=0.431 (0.30–0.50 borderline AD) |
| tan70 | hERG 1 µM | (i) Prodrug motif | FN | 1 | 0.0144 | 0.625 | `CCOC(=O)c1ccc(NC(=O)NNS(=O)(=O)c2cc(OC)ccc2OC)cc1` | SMARTS:carboxylic_ester |
| tan70 | hERG 1 µM | (i) Prodrug motif | FN | 1 | 0.0542 | 0.408 | `C=C(C)CN=C(S)NN=C1c2ccccc2-c2ccccc21` | SMARTS:schiff_base |
| tan70 | hERG 1 µM | (i) Prodrug motif | FN | 1 | 0.0866 | 0.632 | `CCCCCCCCOc1ccccc1C(=O)Nc1ccc(C(=O)OCC[N+](C)(CC)CC)cc1` | SMARTS:carboxylic_ester |
| tan70 | hERG 1 µM | (iii) Label noise | FN | 1 | 0.0753 | 0.621 | `Cc1c(C(=O)NC2CCOC2)oc2c(F)cccc12` | physically inconsistent: 1um=1 but 10um=0 |
| tan70 | hERG 1 µM | (iii) Label noise | FP | 0 | 0.7801 | 0.615 | `CCNC(=O)N1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1` | label=0 but herg_pchembl=7.25 (≥6.5) |
| tan70 | hERG 1 µM | (v) True in-domain model failure | FN | 1 | 0.0341 | 0.578 | `CCC(Sc1nc(-c2ccccc2)c(-c2ccccc2)[nH]1)C(=O)O` | max_tan=0.578 ≥ 0.50, no liability/noise signal |
| tan70 | hERG 1 µM | (v) True in-domain model failure | FN | 1 | 0.0359 | 0.653 | `COc1ccc(C(=O)Nc2nnc(C(F)(F)F)s2)cc1` | max_tan=0.653 ≥ 0.50, no liability/noise signal |
| tan70 | hERG 1 µM | (v) True in-domain model failure | FN | 1 | 0.0612 | 0.506 | `COc1ccc(OC)c(Cc2nc3ccccc3n2CC(=O)Nc2cc(C(C)(C)C)cc(C(C)(C)C)c2)c1` | max_tan=0.506 ≥ 0.50, no liability/noise signal |
| tan70 | hERG 1 µM | Borderline | FN | 1 | 0.0343 | 0.383 | `Cc1ccc(S(=O)(=O)C(C#N)=C2N(C)c3ccccc3N2C)cc1` | max_tan=0.383 (0.30–0.50 borderline AD) |
| tan70 | hERG 1 µM | Borderline | FN | 1 | 0.0431 | 0.473 | `CN1CCN(C(=O)c2ccc(OCc3cccc(Cl)c3)c(Cl)c2)CC1=O` | max_tan=0.473 (0.30–0.50 borderline AD) |
| tan70 | hERG 1 µM | Borderline | FN | 1 | 0.0660 | 0.456 | `O=C1NC(=O)C(Cc2coc3ccc(Cl)cc3c2=O)S1` | max_tan=0.456 (0.30–0.50 borderline AD) |
| tan70 | Nav1.5 | (i) Prodrug motif | FP | 0 | 0.6630 | 0.687 | `Cc1onc(-c2c(Cl)cccc2Cl)c1N(C)C(=O)OCc1ccc(OC(C)C)nc1` | SMARTS:carboxylic_ester|carbamate |
| tan70 | Nav1.5 | (i) Prodrug motif | FN | 1 | 0.3574 | 0.603 | `CC(C)(C)OC(=O)N1CCN(c2cccc3c2cnn3-c2ccccc2F)C1=O` | SMARTS:carboxylic_ester|carbamate |
| tan70 | Nav1.5 | (i) Prodrug motif | FN | 1 | 0.4823 | 0.500 | `CCCC/N=C(/N)NC(=O)c1ccc(C)cc1` | SMARTS:schiff_base |
| tan70 | Nav1.5 | (ii) Multi-target liability | FP | 0 | 0.8912 | 0.595 | `Nc1cncc(Nc2ccc(Oc3ccc(Cl)cc3)cc2)c1` | also blocker=1 in herg_blocker_10um,herg_blocker_1um |
| tan70 | Nav1.5 | (ii) Multi-target liability | FN | 1 | 0.3265 | 0.403 | `Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)nc1Nc1cccc(S(=O)(=O)NC(C)(C)C)c1` | also blocker=1 in herg_blocker_10um,herg_blocker_1um |
| tan70 | Nav1.5 | (ii) Multi-target liability | FP | 0 | 0.6449 | 0.422 | `CCC(=N\O)/C(C)=C/c1ccc(F)cc1` | also blocker=1 in cav12_blocker |
| tan70 | Nav1.5 | (iv) Outside applicability domain | FN | 1 | 0.3325 | 0.262 | `C[C@H]1C[C@@H](N2C(=N)N[C@](C)(c3cccc(Nc4ccc(C5CC5)nc4)c3Cl)CC2=O)CCO1` | max_tan=0.262 < 0.30 |
| tan70 | Nav1.5 | (iv) Outside applicability domain | FP | 0 | 0.6500 | 0.295 | `CCOc1nc2cccc(C(=O)OC(C)OC(=O)OC3CCCCC3)c2n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1` | max_tan=0.295 < 0.30 |
| tan70 | Nav1.5 | (v) True in-domain model failure | FP | 0 | 0.9271 | 0.545 | `O=C(c1ccc(C(F)(F)F)cc1)N1CCC2(CCc3ccccc3O2)CC1` | max_tan=0.545 ≥ 0.50, no liability/noise signal |
| tan70 | Nav1.5 | (v) True in-domain model failure | FP | 0 | 0.9218 | 0.578 | `Cc1nc(C(F)(F)F)n2cc(-c3ccc(OC(F)(F)F)cc3)ccc12` | max_tan=0.578 ≥ 0.50, no liability/noise signal |
| tan70 | Nav1.5 | (v) True in-domain model failure | FP | 0 | 0.9040 | 0.667 | `FC(F)(F)Oc1ccc(-c2ccn3c(C(F)(F)F)nnc3c2)cc1` | max_tan=0.667 ≥ 0.50, no liability/noise signal |
| tan70 | Nav1.5 | Borderline | FP | 0 | 0.7476 | 0.339 | `Cc1cc(C#N)cc(C)c1Oc1nc(Nc2ccc(C#N)cc2)nc(N)c1Br` | max_tan=0.339 (0.30–0.50 borderline AD) |
| tan70 | Nav1.5 | Borderline | FP | 0 | 0.7427 | 0.440 | `FC(F)(F)Oc1ccc(-c2ccn3nccc3c2)cc1` | max_tan=0.440 (0.30–0.50 borderline AD) |
| tan70 | Nav1.5 | Borderline | FP | 0 | 0.7220 | 0.369 | `CSc1c(C#N)c(=O)c2ccc(F)cc2n1-c1c(Cl)cccc1Cl` | max_tan=0.369 (0.30–0.50 borderline AD) |
| tan70 | Cav1.2 | (i) Prodrug motif | FP | 0 | 0.9769 | 0.674 | `CCOC(=O)c1c(C)nc(C)c(C(=O)OCC)c1-c1c(-c2ccccc2)noc1C` | SMARTS:carboxylic_ester |
| tan70 | Cav1.2 | (i) Prodrug motif | FP | 0 | 0.7303 | 0.520 | `CCCCCCOC(=O)[C@H](CCC(=O)NCCC1CCN(Cc2ccccc2)CC1)NC(=O)c1ccccc1` | SMARTS:carboxylic_ester |
| tan70 | Cav1.2 | (i) Prodrug motif | FP | 0 | 0.5745 | 0.530 | `CCOC(=O)CC[C@H](NC(=O)OCc1ccccc1)C(=O)NCCC1CCN(Cc2ccccc2)CC1` | SMARTS:carboxylic_ester|carbamate |
| tan70 | Cav1.2 | (ii) Multi-target liability | FN | 1 | 0.2695 | 0.359 | `Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1` | also blocker=1 in herg_blocker_10um |
| tan70 | Cav1.2 | (ii) Multi-target liability | FP | 0 | 0.6890 | 0.635 | `N#Cc1ccc(Cn2ccc(NC(=O)Cc3ccc(C4(C(F)(F)F)CC4)cc3)n2)nc1` | also blocker=1 in herg_blocker_10um |
| tan70 | Cav1.2 | (ii) Multi-target liability | FN | 1 | 0.3211 | 0.395 | `CC(CC(c1ccccc1)c1ccccc1)NC(C)(C)C` | also blocker=1 in herg_blocker_10um,herg_blocker_1um,nav15_blocker |
| tan70 | Cav1.2 | (iv) Outside applicability domain | FN | 1 | 0.0170 | 0.222 | `C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(S[C@@H]3CN[C@H](CNS(N)(=O)=O)C3)[C@H](C)[C@H]12` | max_tan=0.222 < 0.30 |
| tan70 | Cav1.2 | (iv) Outside applicability domain | FP | 0 | 0.6061 | 0.291 | `Cn1nc(C(=O)OCCCCNC[C@H](O)c2ccc(O)c3[nH]c(=O)ccc23)cc1COc1cccc([C@@H](NC(=O)O[C@H]2CN3CCC2CC3)c2ccccc2)c1` | max_tan=0.291 < 0.30 |
| tan70 | Cav1.2 | (v) True in-domain model failure | FN | 1 | 0.0979 | 0.675 | `FC(F)(F)Cc1nc2c(C(F)(F)F)cc(C(F)(F)F)cc2[nH]1` | max_tan=0.675 ≥ 0.50, no liability/noise signal |
| tan70 | Cav1.2 | (v) True in-domain model failure | FP | 0 | 0.7910 | 0.595 | `CCNc1ccc(C(F)(F)F)c(Nc2c(CC)cccc2CC)n1` | max_tan=0.595 ≥ 0.50, no liability/noise signal |
| tan70 | Cav1.2 | (v) True in-domain model failure | FP | 0 | 0.7578 | 0.607 | `CN(CC(N)=O)[C@H](COCc1cc(C(F)(F)F)cc(C(F)(F)F)c1)c1ccccc1` | max_tan=0.607 ≥ 0.50, no liability/noise signal |
| tan70 | Cav1.2 | Borderline | FN | 1 | 0.0306 | 0.423 | `CC(C)CC(CN)CC(=O)O` | max_tan=0.423 (0.30–0.50 borderline AD) |
| tan70 | Cav1.2 | Borderline | FP | 0 | 0.8862 | 0.389 | `CC1=C(C#N)C(c2ccc3[nH]nc(C)c3c2)C(C#N)=C(C)N1` | max_tan=0.389 (0.30–0.50 borderline AD) |
| tan70 | Cav1.2 | Borderline | FN | 1 | 0.2491 | 0.422 | `CCC(=N\O)/C(C)=C/c1ccc(F)cc1` | max_tan=0.422 (0.30–0.50 borderline AD) |
| tan70 | IKs | (v) True in-domain model failure | FN | 1 | 0.1760 | 0.643 | `c1cncc(C(Cc2ccccc2-c2cncc3ccccc23)c2cccnc2)c1` | max_tan=0.643 ≥ 0.50, no liability/noise signal |
| tan70 | IKs | (v) True in-domain model failure | FP | 0 | 0.6421 | 0.509 | `OC(Cc1cccnc1-c1cccc(C(F)(F)F)c1)(c1cccnc1F)c1cccnc1F` | max_tan=0.509 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 10 µM | (i) Prodrug motif | FP | 0 | 0.9171 | 0.562 | `CC1(C)N=C(N)N=C(N)N1c1ccc(Oc2ccc(Cl)cc2)c(Cl)c1` | SMARTS:schiff_base |
| tan60 | hERG 10 µM | (i) Prodrug motif | FP | 0 | 0.8979 | 0.587 | `CCOC(=O)c1cnc(NC2CCN(Cc3ccc(C(=O)NC)cc3)CC2)nc1Oc1c(C)cc(-c2ccc(C#N)cc2)cc1C` | SMARTS:carboxylic_ester |
| tan60 | hERG 10 µM | (i) Prodrug motif | FN | 1 | 0.1102 | 0.590 | `Cc1cc(SCC(=O)OCC(=O)NCC2CCCO2)c(C)cc1Br` | SMARTS:carboxylic_ester |
| tan60 | hERG 10 µM | (iv) Outside applicability domain | FP | 0 | 0.9708 | 0.053 | `[Hg+2]` | max_tan=0.053 < 0.30 |
| tan60 | hERG 10 µM | (iv) Outside applicability domain | FP | 0 | 0.9696 | 0.067 | `[Cd+2]` | max_tan=0.067 < 0.30 |
| tan60 | hERG 10 µM | (v) True in-domain model failure | FN | 1 | 0.0538 | 0.583 | `CCCN(Cc1ccccc1)C(S)=Nc1cccc(C)c1C` | max_tan=0.583 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 10 µM | (v) True in-domain model failure | FN | 1 | 0.0709 | 0.567 | `CCCN(C(=O)CN1C(=O)NC2(CCCCCC2)C1=O)C1CCS(=O)(=O)C1` | max_tan=0.567 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 10 µM | (v) True in-domain model failure | FP | 0 | 0.9264 | 0.525 | `C[n+]1c(-c2ccccc2)cc(CCCCc2cc(-c3ccccc3)[n+](C)c(-c3ccccc3)c2)cc1-c1ccccc1.[O-][Cl+3]([O-])([O-])O` | max_tan=0.525 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 10 µM | Borderline | FN | 1 | 0.0908 | 0.431 | `CC1=CC(=O)C(C(C)C)=CC1=NC(=O)c1ccc(C)cc1` | max_tan=0.431 (0.30–0.50 borderline AD) |
| tan60 | hERG 10 µM | Borderline | FP | 0 | 0.9014 | 0.420 | `O=S1(=O)N(CCN2CC=C(c3c[nH]c4cc(F)ccc34)CC2)c2cccc3c2N1CCC3` | max_tan=0.420 (0.30–0.50 borderline AD) |
| tan60 | hERG 10 µM | Borderline | FP | 0 | 0.8979 | 0.375 | `CCN(CC)CCCC(C)N=c1cc(C=Cc2ccccc2Cl)[nH]c2cc(Cl)ccc12` | max_tan=0.375 (0.30–0.50 borderline AD) |
| tan60 | hERG 1 µM | (i) Prodrug motif | FN | 1 | 0.0937 | 0.368 | `CC(=O)NC(=O)C1CCC(C2N=CC(c3cccc(Br)c3)=N2)CC1` | SMARTS:schiff_base |
| tan60 | hERG 1 µM | (i) Prodrug motif | FP | 0 | 0.8168 | 0.562 | `CC1(C)N=C(N)N=C(N)N1c1ccc(Oc2ccc(Cl)cc2)c(Cl)c1` | SMARTS:schiff_base |
| tan60 | hERG 1 µM | (i) Prodrug motif | FN | 1 | 0.2022 | 0.511 | `CC/C=C(\C(=O)N[C@@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(COC(N)=O)CS[C@H]12)c1csc(N)n1` | SMARTS:carboxylic_ester|carbamate |
| tan60 | hERG 1 µM | (ii) Multi-target liability | FP | 0 | 0.7775 | 0.447 | `Oc1cccc(N2[C@@H]3CC[C@H]2CN(CCCc2ccccc2)C3)c1` | also blocker=1 in nav15_blocker |
| tan60 | hERG 1 µM | (iii) Label noise | FN | 1 | 0.0948 | 0.545 | `c1ccc(OCc2cccc(COc3ccccc3)n2)cc1` | physically inconsistent: 1um=1 but 10um=0 |
| tan60 | hERG 1 µM | (iii) Label noise | FN | 1 | 0.2021 | 0.537 | `Cc1ccc(Cc2cccc(COc3ccc(C)cc3)n2)cc1` | physically inconsistent: 1um=1 but 10um=0 |
| tan60 | hERG 1 µM | (iv) Outside applicability domain | FP | 0 | 0.8053 | 0.067 | `[Cd+2]` | max_tan=0.067 < 0.30 |
| tan60 | hERG 1 µM | (iv) Outside applicability domain | FP | 0 | 0.8052 | 0.053 | `[Hg+2]` | max_tan=0.053 < 0.30 |
| tan60 | hERG 1 µM | (v) True in-domain model failure | FN | 1 | 0.0307 | 0.578 | `CCC(Sc1nc(-c2ccccc2)c(-c2ccccc2)[nH]1)C(=O)O` | max_tan=0.578 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 1 µM | (v) True in-domain model failure | FN | 1 | 0.0551 | 0.588 | `Cc1ccsc1/C=N/N(C)c1cnc2ccccc2n1` | max_tan=0.588 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 1 µM | (v) True in-domain model failure | FN | 1 | 0.0600 | 0.597 | `COc1ccc(C(NC(=O)c2cnc(-n3cccn3)nc2O)c2ccc(OC)cc2)cc1` | max_tan=0.597 ≥ 0.50, no liability/noise signal |
| tan60 | hERG 1 µM | Borderline | FN | 1 | 0.0399 | 0.333 | `N/N=c1\[nH][nH]/c(=N\N)c2ccccc12` | max_tan=0.333 (0.30–0.50 borderline AD) |
| tan60 | hERG 1 µM | Borderline | FN | 1 | 0.0887 | 0.456 | `O=C1NC(=O)C(Cc2coc3ccc(Cl)cc3c2=O)S1` | max_tan=0.456 (0.30–0.50 borderline AD) |
| tan60 | hERG 1 µM | Borderline | FP | 0 | 0.8753 | 0.460 | `Clc1ccc(Cn2c(/C=C/C3CCN(CC4CCCCC4)C3)nc3cc(Br)ccc32)cc1` | max_tan=0.460 (0.30–0.50 borderline AD) |
| tan60 | Nav1.5 | (ii) Multi-target liability | FN | 1 | 0.3777 | 0.403 | `Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)nc1Nc1cccc(S(=O)(=O)NC(C)(C)C)c1` | also blocker=1 in herg_blocker_10um,herg_blocker_1um |
| tan60 | Nav1.5 | (iv) Outside applicability domain | FN | 1 | 0.1805 | 0.232 | `N=C1N[C@@]23[C@H]([C@H]4NC(=N)N2C[C@](OS(=O)(=O)O)(C[C@H]2[C@@H](CO)CON2C4=O)C3(O)O)N1COC(=O)NO` | max_tan=0.232 < 0.30 |
| tan60 | Nav1.5 | (v) True in-domain model failure | FP | 0 | 0.8212 | 0.578 | `Cc1nc(C(F)(F)F)n2cc(-c3ccc(OC(F)(F)F)cc3)ccc12` | max_tan=0.578 ≥ 0.50, no liability/noise signal |
| tan60 | Nav1.5 | (v) True in-domain model failure | FP | 0 | 0.7112 | 0.545 | `FC(F)(F)Oc1ccc(-c2ccc3nc(C(F)(F)F)nn3c2)cc1` | max_tan=0.545 ≥ 0.50, no liability/noise signal |
| tan60 | Nav1.5 | (v) True in-domain model failure | FP | 0 | 0.6607 | 0.595 | `CCNc1ccc(C(F)(F)F)c(Nc2c(CC)cccc2CC)n1` | max_tan=0.595 ≥ 0.50, no liability/noise signal |
| tan60 | Nav1.5 | Borderline | FP | 0 | 0.6555 | 0.489 | `FC(F)(F)Oc1ccc(-c2ccn3nccc3c2)cc1` | max_tan=0.489 (0.30–0.50 borderline AD) |
| tan60 | Nav1.5 | Borderline | FP | 0 | 0.6545 | 0.458 | `N=C(N)NCCC[C@@H]1NC(=O)[C@H](CO)NC(=O)[C@H](Cc2c[nH]cn2)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H]2CSSC[C@@H]3NC(=O)[C@@H](N)CSSC[C@H](NC1=O)C(=O)N[C@H](C(=O)O)CSSC[C@H](NC(=O)[C@H](CC(N)=O)NC3=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](Cc1c[nH]c3ccccc13)C(=O)N2` | max_tan=0.458 (0.30–0.50 borderline AD) |
| tan60 | Nav1.5 | Borderline | FP | 0 | 0.6082 | 0.369 | `CSc1c(C#N)c(=O)c2ccc(F)cc2n1-c1c(Cl)cccc1Cl` | max_tan=0.369 (0.30–0.50 borderline AD) |
| tan60 | Cav1.2 | (i) Prodrug motif | FP | 0 | 0.5098 | 0.520 | `CCCCCCOC(=O)[C@H](CCC(=O)NCCC1CCN(Cc2ccccc2)CC1)NC(=O)c1ccccc1` | SMARTS:carboxylic_ester |
| tan60 | Cav1.2 | (iv) Outside applicability domain | FP | 0 | 0.6015 | 0.291 | `Cn1nc(C(=O)OCCCCNC[C@H](O)c2ccc(O)c3[nH]c(=O)ccc23)cc1COc1cccc([C@@H](NC(=O)O[C@H]2CN3CCC2CC3)c2ccccc2)c1` | max_tan=0.291 < 0.30 |
| tan60 | Cav1.2 | (v) True in-domain model failure | FN | 1 | 0.1687 | 0.522 | `COc1ccc([C@H](CN(C)C)[Si]2(O)CCCCC2)cc1` | max_tan=0.522 ≥ 0.50, no liability/noise signal |
| tan60 | Cav1.2 | (v) True in-domain model failure | FN | 1 | 0.3826 | 0.571 | `COc1ccc(CN2CCN(C(=O)CCn3cc(C)c4ccccc43)CC2)cc1OCc1ccccc1` | max_tan=0.571 ≥ 0.50, no liability/noise signal |
| tan60 | Cav1.2 | (v) True in-domain model failure | FP | 0 | 0.5937 | 0.595 | `CCNc1ccc(C(F)(F)F)c(Nc2c(CC)cccc2CC)n1` | max_tan=0.595 ≥ 0.50, no liability/noise signal |
| tan60 | Cav1.2 | Borderline | FN | 1 | 0.2276 | 0.488 | `C[C@H]1CN(C[C@H](Cc2ccccc2)C(=O)NCC(=O)O)CC[C@@]1(C)c1cccc(O)c1` | max_tan=0.488 (0.30–0.50 borderline AD) |
| tan60 | Cav1.2 | Borderline | FP | 0 | 0.7213 | 0.438 | `CC1=C(C#N)C(c2ccc3[nH]nc(C)c3c2)C(C#N)=C(C)N1` | max_tan=0.438 (0.30–0.50 borderline AD) |
| tan60 | Cav1.2 | Borderline | FN | 1 | 0.4084 | 0.405 | `CCCOc1ccc(CCc2nc3cc(-c4c(C)noc4C)ccc3n2CCN2CCOCC2)cc1` | max_tan=0.405 (0.30–0.50 borderline AD) |
| tan60 | IKs | Borderline | FN | 1 | 0.2608 | 0.438 | `CN(C)C(=O)c1cccc(-c2ncccc2CC(c2ccnc(N)n2)c2cccnc2F)c1` | max_tan=0.438 (0.30–0.50 borderline AD) |

## Bucket counts (representatives only)

Reported counts here are *representatives in this table* (3 per (head, bucket)
where data was available). The aggregate counts cited in the manuscript
(269 false predictions total: 41% true in-domain failure, 37% borderline,
12% prodrug, 5% multi-target, 3% out-of-AD, 2% label noise) include all
examined false predictions across both splits and all heads — not just the
few drawn here as representatives.

| split | head | i_prodrug | ii_multi_target | iii_label_noise | iv_AD | v_true_failure | borderline |
| :--: | :-- | --: | --: | --: | --: | --: | --: |
| tan70 | hERG 10 µM | 3 | 0 | 2 | 0 | 3 | 3 |
| tan70 | hERG 1 µM | 3 | 0 | 2 | 0 | 3 | 3 |
| tan70 | Nav1.5 | 3 | 3 | 0 | 2 | 3 | 3 |
| tan70 | Cav1.2 | 3 | 3 | 0 | 2 | 3 | 3 |
| tan70 | IKs | 0 | 0 | 0 | 0 | 2 | 0 |
| tan60 | hERG 10 µM | 3 | 0 | 0 | 2 | 3 | 3 |
| tan60 | hERG 1 µM | 3 | 1 | 2 | 2 | 3 | 3 |
| tan60 | Nav1.5 | 0 | 1 | 0 | 1 | 3 | 3 |
| tan60 | Cav1.2 | 1 | 0 | 0 | 1 | 3 | 3 |
| tan60 | IKs | 0 | 0 | 0 | 0 | 0 | 1 |

## Schema

| column | description |
| :-- | :-- |
| `split` | tan70 or tan60 |
| `head` | binary head this compound was flagged on |
| `head_id` | machine-readable head identifier |
| `bucket` | machine-readable failure-mode bucket id |
| `bucket_label` | human-readable bucket label (paper numbering) |
| `kind` | `FP` (false positive) or `FN` (false negative) |
| `smiles` | canonical SMILES |
| `inchikey` | 27-character standard InChI-key (computed via RDKit) |
| `true_label` | ground-truth label (0 or 1) |
| `pred_prob` | deployed 5-seed ensemble post-sigmoid probability |
| `max_tan_to_train` | maximum Morgan-r=2 2048-bit Tanimoto to any training compound |
| `rationale` | short human-readable rationale for the bucket assignment |
