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

## `splits/` — Tanimoto-controlled splits

Two versions are shipped side-by-side; see `splits/CHANGELOG.md`.

| File | Version | Notes |
| --- | --- | --- |
| `splits/tan70.csv`, `splits/tan60.csv` | **v1.0** | The splits as used in the bioRxiv preprint. Carry 12 (tan70) and 15 (tan60) train→val cross-fold edges in the hydroxymethyl-terfenadine cluster — the audit catches them. Retained for reproducibility of the v1.0 numbers. |
| `splits/tan70_v1_1.csv`, `splits/tan60_v1_1.csv` | **v1.1** | Audit-clean. The 3 HMT analogs of the cardiac-cliff anchors are routed to `val`. Test fold is identical to v1.0. **Use this for any new work.** |

Schema (same for both versions):

| column | dtype | notes |
| --- | --- | --- |
| `row_idx` | int | canonical position (joins to `compounds.csv`) |
| `inchikey` | str | duplicated here so the file is self-contained |
| `fold` | str | one of `train`, `val`, `test`, or `excluded` |

Construction:

- **tan70** (primary): Morgan radius-2 2048-bit Tanimoto ≥ 0.70 forbids
  cross-fold edges. Terfenadine and fexofenadine are force-routed to `val`;
  v1.1 additionally routes the 2 HMT analogs in their ≥-cutoff neighbourhood.
- **tan60** (stricter): same construction at Tanimoto ≥ 0.60. 1 compound is
  marked `excluded` (no fold could be assigned without violating the
  cutoff). v1.1 additionally routes the 3 HMT analogs to val.

Fold counts:

| version | split | train | val | test | excluded |
| --- | --- | --- | --- | --- | --- |
| v1.0 | tan70 | 241,792 | 46,326 | 46,326 | 0 |
| v1.0 | tan60 | 306,665 | 13,889 | 13,889 | 1 |
| **v1.1** | tan70 | 241,790 | 46,328 | 46,326 | 0 |
| **v1.1** | tan60 | 306,662 | 13,892 | 13,889 | 1 |

To verify a split is audit-clean:

```bash
python scripts/audit_tanimoto_leak.py --split tan70_v1_1
python scripts/audit_tanimoto_leak.py --split tan60_v1_1
```

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
| `supplementary/table_s9_curation_sensitivity.csv` | Table S9 — curation sensitivity, strict vs full. AUC + MCC for the 5 binary heads and Pearson r + RMSE for the 3 regression heads, across three curation policies (full pharmacology-aware / exact-only / exact-only-strict) on both splits. Includes per-channel training-pool label-count erosion. |
| `supplementary/table_s9_curation_sensitivity.md` | Same content rendered as Markdown |
| `supplementary/table_s9_curation_sensitivity.json` | Same content as JSON |
| `supplementary/figure_s1_herg_calibration.pdf` | Figure S1 — reliability / calibration curves for hERG across applicability-domain bins on tan70. 2 rows (hERG 10 µM, hERG 1 µM) × 3 cols (max-Tanimoto-to-train bins: 0.0–0.3, 0.3–0.5, 0.5–0.7). Each panel: mean predicted probability vs observed positive rate (deciles), with diagonal reference and ECE annotation. |
| `supplementary/figure_s1_herg_calibration.png` | Same figure rasterised at 300 dpi |
| `supplementary/figure_s1_herg_calibration.json` | Underlying decile data (`p_lo`, `p_hi`, `p_mean`, `obs_pos_rate`, `n`) plus per-panel total *n* and ECE |
| `supplementary/note_s1_yrand_mechanism.md` | Note S1 — Y-randomization mechanism + per-MW-stratum analysis explaining the elevated unstratified hERG Y-randomized AUC (~0.70) as a molecular-weight prevalence confound. Includes the unconfounded controls (regression *r* and MCC) collapsing to ~0. |
| `supplementary/note_s1_yrand_data.json` | Underlying data behind Note S1 — V6 and Stage 2 Y-randomized overall AUCs, per-MW-bin metrics, seed-42 unconfounded controls, prevalence-controlled subsample |
| `supplementary/note_s2_cliff_curation.md` | Note S2 — prose documenting the two-stage activity-cliff curation pipeline (manual literature curation + automated Tanimoto leak-prevention filter), with the 25-source bibliography and the 30-`pair_id` composition table |
| `supplementary/note_s2_pair_id_composition.csv` | Per-`pair_id` table: therapeutic class, blocker members, safer members, sources, ranking-pair eligibility per split |
| `supplementary/note_s2_per_split_manifest.csv` | One row per cliff compound × split with include / exclude status, literature IC50, and herg_1um / herg_10um block flags |
| `supplementary/note_s2_source_bibliography.csv` | 25 source citation keys with per-source compound counts and citation stubs (full citations to be completed by author at proof stage) |
| `supplementary/note_s2_cliff_set_tan70.csv` | Per-split filtered cliff manifest for tan70 (48 compounds surviving Stage B of the curation) |
| `supplementary/note_s2_cliff_set_tan60.csv` | Per-split filtered cliff manifest for tan60 (51 compounds surviving Stage B) |
| `supplementary/note_s3_v1_1_audit_correction.md` | Note S3 — audit-driven correction: documents the 12 (tan70) / 15 (tan60) cliff-cluster train ↔ val edges caught by `scripts/audit_tanimoto_leak.py`, the v1.1 patch (2 compounds moved to val on tan70; 3 on tan60), updated fold counts, the v1.1 retrain on tan70_v1_1, headline test-fold metric shifts, and the terfenadine/fexofenadine case-study re-evaluation (qualitative narrative preserved). Test fold is unchanged in both splits. |
| `supplementary/note_s3_v1_1_test_metrics.json` | Per-head v1.1 test-fold metrics on tan70_v1_1 (5-seed ensemble, Stage 1 + Stage 2, plus per-seed min/max). |
| `supplementary/note_s3_v1_1_case_study.json` | Per-compound v1.0 vs v1.1 predictions for terfenadine + fexofenadine: pIC50 with inter-seed σ, classification CO at 10 µM and 1 µM, per-compound Δ, and the predicted cliff comparison. |
| `comparators/ctoxpred2_tan70_predictions.csv` | CToxPred2 per-compound block probabilities on the tan70 test fold — input to the reverse-leak audit and head-to-head comparison (Tables 3 / 3b / S2 / S3 / Figure 4) |
| `comparators/cardiogenai_tan70_predictions.csv` | CardioGenAI per-compound block probabilities on the tan70 test fold |
| `comparators/README.md` | Schema, citations, coverage notes, and rationale for shipping these CSVs alongside the data deposit |

All paper-promised supplementary materials are now in this directory.See
the manuscript's *Additional files* section for the original list and the
context in which each item is cited.

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
