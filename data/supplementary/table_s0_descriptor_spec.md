# Table S0 — Descriptor specification

Full 20-descriptor specification used in the 6164-dimensional chemical
input block (3 × 2048-bit fingerprints + this 20-d descriptor block),
with per-column non-zero fractions and summary statistics computed over
the 334,444-compound production cache.

Group A: thirteen RDKit / Gasteiger descriptors (rows 0–12).
Group B: seven ionization-derived descriptors — MolGpKa-predicted pKa
values and Henderson–Hasselbalch closed-form logD and microstate fractions
at pH 7.4 (rows 13–19).

| # | column_id | name | unit | non-zero fraction | mean | SD | min | median | max |
| -: | :-- | :-- | :--: | --: | --: | --: | --: | --: | --: |
| 0 | `mw` | Molecular weight | g/mol | 1.0000 | 367.4 | 123.8 | 36.46 | 361.5 | 4272 |
| 1 | `clogp` | Crippen clogP | log10 | 1.0000 | 3.038 | 1.32 | -17.64 | 3.062 | 15.16 |
| 2 | `tpsa` | Topological polar surface area | Å² | 0.9994 | 86.94 | 53.17 | 0 | 83.75 | 1896 |
| 3 | `hbd` | Hydrogen-bond donor count | count | 0.7795 | 1.208 | 1.633 | 0 | 1 | 60 |
| 4 | `hba` | Hydrogen-bond acceptor count | count | 0.9988 | 4.699 | 2.201 | 0 | 5 | 63 |
| 5 | `rotatable_bonds` | Rotatable bond count | count | 0.9917 | 5.078 | 2.91 | 0 | 5 | 85 |
| 6 | `aromatic_rings` | Aromatic ring count | count | 0.9824 | 2.306 | 0.9428 | 0 | 2 | 12 |
| 7 | `heavy_atom_count` | Heavy atom count | count | 1.0000 | 25.69 | 8.643 | 1 | 25 | 295 |
| 8 | `gasteiger_mean` | Gasteiger partial charge mean | e | 0.9998 | -0.04613 | 0.01483 | -1 | -0.04593 | 6 |
| 9 | `gasteiger_max` | Gasteiger partial charge max | e | 0.9998 | 0.2744 | 0.0733 | -1 | 0.2655 | 6 |
| 10 | `gasteiger_min` | Gasteiger partial charge min | e | 0.9998 | -0.4227 | 0.07752 | -1 | -0.4553 | 6 |
| 11 | `gasteiger_std` | Gasteiger partial charge SD | e | 0.9998 | 0.1682 | 0.03017 | 0 | 0.1683 | 1.18 |
| 12 | `max_positive_n` | Max positive Gasteiger charge on N | e | 0.9791 | -0.1995 | 0.1045 | -0.6758 | -0.2111 | 0.4235 |
| 13 | `pka_acidic` | Most acidic predicted pKa | pKa | 1.0000 | 11.21 | 3.581 | -5.135 | 12.85 | 14.26 |
| 14 | `pka_basic` | Most basic predicted pKa | pKa | 0.8319 | 3.935 | 2.694 | 0 | 3.93 | 20.2 |
| 15 | `logd_7_4` | logD at pH 7.4 | log10 | 1.0000 | 2.327 | 2.131 | -25.05 | 2.701 | 15.16 |
| 16 | `frac_cation` | Cation microstate fraction @ pH 7.4 | fraction | 1.0000 | 0.1034 | 0.2672 | 3.299e-20 | 0.0001613 | 1 |
| 17 | `frac_anion` | Anion microstate fraction @ pH 7.4 | fraction | 1.0000 | 0.1313 | 0.3136 | 4.005e-20 | 2.87e-06 | 1 |
| 18 | `frac_zwitterion` | Zwitterion microstate fraction @ pH 7.4 | fraction | 1.0000 | 0.02304 | 0.1373 | 7.46e-15 | 1.63e-08 | 1 |
| 19 | `frac_neutral` | Neutral microstate fraction @ pH 7.4 | fraction | 1.0000 | 0.7423 | 0.3974 | 1.496e-17 | 0.9965 | 1 |

## Definitions and sources

| column_id | definition | source |
| :-- | :-- | :-- |
| `mw` | Average molecular weight from canonical SMILES | RDKit Descriptors.MolWt |
| `clogp` | Wildman–Crippen octanol/water partition coefficient (calculated) | RDKit Crippen.MolLogP |
| `tpsa` | Sum of polar atom surface contributions (Ertl method) | RDKit Descriptors.TPSA |
| `hbd` | Number of OH and NH donors (Lipinski definition) | RDKit Lipinski.NumHDonors |
| `hba` | Number of N and O acceptors (Lipinski definition) | RDKit Lipinski.NumHAcceptors |
| `rotatable_bonds` | Number of rotatable single bonds (Lipinski definition) | RDKit Lipinski.NumRotatableBonds |
| `aromatic_rings` | Number of aromatic rings in the molecule | RDKit Lipinski.NumAromaticRings |
| `heavy_atom_count` | Total number of non-hydrogen atoms | RDKit Descriptors.HeavyAtomCount |
| `gasteiger_mean` | Mean of per-atom Gasteiger partial charges | RDKit AllChem.ComputeGasteigerCharges |
| `gasteiger_max` | Maximum of per-atom Gasteiger partial charges | RDKit AllChem.ComputeGasteigerCharges |
| `gasteiger_min` | Minimum of per-atom Gasteiger partial charges | RDKit AllChem.ComputeGasteigerCharges |
| `gasteiger_std` | Standard deviation of per-atom Gasteiger partial charges | RDKit AllChem.ComputeGasteigerCharges |
| `max_positive_n` | Largest Gasteiger partial charge on any nitrogen atom; 0.0 if no N is present | RDKit AllChem.ComputeGasteigerCharges restricted to N atoms |
| `pka_acidic` | Lowest predicted pKa across acidic sites; sentinel 14.0 if no acidic site detected | MolGpKa acid model [Pan et al. 2021] |
| `pka_basic` | Highest predicted pKa across basic sites; sentinel 0.0 if no basic site detected (~19% of pool) | MolGpKa base model [Pan et al. 2021] |
| `logd_7_4` | Effective octanol/water partition coefficient at pH 7.4; Henderson–Hasselbalch independent-site closed form using clogP + predicted pKa values, neutral-microstate-only partitioning | Closed-form HH from clogP, pka_acidic, pka_basic |
| `frac_cation` | Fraction of population in the net-positive microstate at pH 7.4 | Henderson–Hasselbalch independent-site model |
| `frac_anion` | Fraction of population in the net-negative microstate at pH 7.4 | Henderson–Hasselbalch independent-site model |
| `frac_zwitterion` | Fraction of population in the simultaneously protonated-acid and deprotonated-base microstate at pH 7.4 | Henderson–Hasselbalch independent-site model |
| `frac_neutral` | Fraction of population in the unionized microstate at pH 7.4 | Henderson–Hasselbalch independent-site model |

## Notes

- Descriptor columns are z-scored per-column at training time using means and standard deviations fitted on the **training fold only**; the table here reports the **unscaled** statistics across the full 334,444-row pool.
- Zero-variance columns are left unscaled at training time. None of the 20 columns is zero-variance over the production cache.
- `max_positive_n` is 0.0 for compounds with no nitrogen atom; `pka_acidic` is 14.0 and `pka_basic` is 0.0 for compounds without a detectable acidic / basic site (~19% of the pool carries no detectable basic site).
- `frac_cation + frac_anion + frac_zwitterion + frac_neutral = 1.0` to within numerical noise (verified across the production cache).
