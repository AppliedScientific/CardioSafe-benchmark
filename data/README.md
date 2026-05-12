# Data

All files in this tree are released under
[CC BY 4.0](../LICENSE-DATA). The benchmark's row order is canonical:
the integer in the `row_idx` column points at the same compound across every
file in this tree.

## `compounds/compounds.csv` — 334,444 × 3

The full compound manifest. One row per curated molecule.

| column | dtype | notes |
| --- | --- | --- |
| `row_idx` | int | canonical 0-indexed position in the dataset |
| `smiles` | str | canonical SMILES (RDKit 2025.09.6, salt-stripped + neutralized) |
| `inchikey` | str | 27-character standard InChI-key |

Compounds were curated from ChEMBL 36 (source dump SHA-256
`b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`) and the
hERG Central primary screen. Standardisation: RDKit parse → salt strip →
charge neutralization → sanitization → canonical-SMILES regeneration.
InChI-keys are the 27-character standard InChI-key produced by RDKit's
`MolToInchiKey`.

## `splits/tan70.csv` and `splits/tan60.csv` — 334,444 × 3 each

The Tanimoto-controlled splits used in the paper.

| column | dtype | notes |
| --- | --- | --- |
| `row_idx` | int | canonical position (joins to `compounds.csv`) |
| `inchikey` | str | duplicated here so the file is self-contained |
| `fold` | str | one of `train`, `val`, `test`, or `excluded` |

- **tan70** (primary): Morgan radius-2 2048-bit Tanimoto ≥ 0.70 forbids
  cross-fold edges. Terfenadine and fexofenadine are force-excluded from the
  `train` fold and routed to `val`.
- **tan60** (stricter): same construction at Tanimoto ≥ 0.60. 1 compound is
  marked `excluded` (no fold could be assigned without violating the
  cutoff).

Fold counts:

| split | train | val | test | excluded |
| --- | --- | --- | --- | --- |
| tan70 | 241,792 | 46,326 | 46,326 | 0 |
| tan60 | 306,665 | 13,889 | 13,889 | 1 |

## `labels/labels_v1.csv` — 334,444 × 10

The curated multi-task label matrix. One row per compound; **most cells are
empty** because each compound was only assayed against the channels it has
data for.

| column | dtype | notes |
| --- | --- | --- |
| `row_idx` | int | canonical position (joins to `compounds.csv`) |
| `inchikey` | str | duplicated here so the file is self-contained |
| `herg_pchembl` | float | hERG pIC50 (regression, `-log10(IC50/M)`) |
| `herg_blocker_10um` | 0/1 | hERG blocker @ 10 µM |
| `herg_blocker_1um`  | 0/1 | hERG blocker @ 1 µM |
| `nav15_pchembl` | float | Nav1.5 pIC50 |
| `nav15_blocker` | 0/1 | Nav1.5 blocker @ 10 µM |
| `cav12_pchembl` | float | Cav1.2 pIC50 |
| `cav12_blocker` | 0/1 | Cav1.2 blocker @ 10 µM |
| `iks_blocker`   | 0/1 | IKs blocker @ 10 µM (exploratory; n = 115) |

Per-channel coverage (primary binary label column):

| channel | n labelled | n blockers |
| --- | --- | --- |
| hERG   (10 µM)  | 331,127 | 11,881 |
| Nav1.5 (10 µM)  | 3,160   | 1,240  |
| Cav1.2 (10 µM)  | 1,138   | 548    |
| IKs    (10 µM)  | 115     | 30     |

Curation policy is **pharmacology-aware**: exact `pchembl_value` records,
censored relations (`>`, `>=`, `<`, `<=` in nM) converted via
`pChEMBL = 9 − log10(nM)`, and inhibition-percentage votes with assay-parsed
test concentration. hERG is augmented with the hERG Central primary screen.
Full provenance, ChEMBL 36 source dump SHA-256, and per-column non-NaN
counts are in `labels/MANIFEST.json`.

## `supplementary/` — paper supplementary tables, notes, and figures

The supplementary materials promised in the *Additional files* section of the
manuscript live here. As of the current release:

| file | what |
| --- | --- |
| `supplementary/table_s0_descriptor_spec.csv` | Table S0 — 20-descriptor specification with per-column non-zero fractions and summary statistics over the 334,444-compound production cache |
| `supplementary/table_s0_descriptor_spec.md` | Same content as a rendered Markdown table |
| `supplementary/table_s0_summary.json` | Same content as a JSON dump (machine-friendly, includes per-descriptor definitions) |
| `supplementary/table_s1_confusion_matrices.csv` | Table S1 — per-head confusion matrices (TP / FP / TN / FN + derived metrics) for the deployed 5-seed ensemble at threshold 0.5, on both tan70 and tan60 test folds |
| `supplementary/table_s1_confusion_matrices.md` | Same content rendered as Markdown |
| `supplementary/table_s1_confusion_matrices.json` | Same content as JSON |
| `supplementary/table_s2_comparator_pre_de_leak.csv` | Table S2 — comparator panel on tan70 *pre-de-leak* (expanded Table 3): CardioSafe vs CToxPred2 and CardioGenAI, on 4 channels (hERG 10 µM, hERG 1 µM, Nav1.5, Cav1.2). Paired-bootstrap Δ-AUC and Δ-MCC with 95% CIs (B = 1000, seed 20260428). |
| `supplementary/table_s2_comparator_pre_de_leak.md` | Same content rendered as Markdown |
| `supplementary/table_s3_comparator_post_de_leak.csv` | Table S3 — same as S2 but *post-de-leak* (expanded Table 3b): comparator-train compounds at Tanimoto ≥ 0.99 removed; reports `n_total`, `n_leak`, `n_kept`. |
| `supplementary/table_s3_comparator_post_de_leak.md` | Same content rendered as Markdown |
| `supplementary/tables_s2_s3_paired_bootstrap.json` | Combined structured JSON underlying S2 + S3 (channels × comparators × pre/post). |
| `supplementary/table_s5_drug_panel_tan60.csv` | Table S5 — the 11-drug case-study panel from main-text Table 5 (tan70), evaluated on the stricter tan60 split. Same schema and same drugs. |
| `supplementary/table_s5_drug_panel_tan60.md` | Same content rendered as Markdown |
| `supplementary/table_s5_drug_panel_tan60.json` | Same content as JSON |
| `supplementary/table_s6_failure_modes.csv` | Table S6 — 94 representative SMILES drawn from the top-confident false predictions across 5 binary heads × 2 splits, classified into 6 failure-mode buckets (prodrug, multi-target, label noise, out-of-AD, true in-domain failure, borderline) with rationale per compound |
| `supplementary/table_s6_failure_modes.md` | Same content rendered as Markdown |
| `supplementary/table_s6_failure_modes.json` | Same content as JSON |
| `supplementary/table_s7_ad_per_bin.csv` | Table S7 — full applicability-domain per-bin metrics. Per (split, head, Tanimoto-to-train bin) triple: n, prevalence, AUC and MCC with bootstrap 95% CIs (B = 1000, seed 20260428). |
| `supplementary/table_s7_ad_per_bin.md` | Same content rendered as Markdown |
| `supplementary/table_s7_ad_per_bin.json` | Same content as JSON |
| `supplementary/table_s8_l1000_threshold_sweep.csv` | Table S8 — L1000 threshold sensitivity sweep: how headline AUC and MCC change when the L1000 encoder's co-expression edge-density threshold *r* moves from the deployed default (0.40) to 0.30 or 0.50, on all 5 heads × 2 splits. Includes per-seed AUC variance and confusion-matrix entries per cell. |
| `supplementary/table_s8_l1000_threshold_sweep.md` | Same content rendered as Markdown, plus the predicted-output graph density vs threshold table that supports the manuscript's "29× denser than training graph" claim |
| `supplementary/table_s8_l1000_threshold_sweep.json` | Same content as JSON |

Notes S1, S2 and Table S9 and Figure S1 will be added in subsequent
releases. Seethe manuscript's *Additional files* section for the full list.

## Loading example

```python
import pandas as pd

compounds = pd.read_csv("data/compounds/compounds.csv")
tan70     = pd.read_csv("data/splits/tan70.csv")
labels    = pd.read_csv("data/labels/labels_v1.csv")

# hERG 10 µM test set: features × ground truth × split fold
test_herg_10um = (
    labels[["row_idx", "herg_blocker_10um"]]
        .dropna()
        .merge(tan70.query("fold == 'test'"), on="row_idx")
        .merge(compounds[["row_idx", "smiles"]], on="row_idx")
)
print(len(test_herg_10um))   # ~46,120 — matches Table 2 of the paper
```
