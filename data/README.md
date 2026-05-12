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

Notes S1, S2 and Tables S1–S9 (no S4) and Figure S1 will be added in
subsequent releases. Seethe manuscript's *Additional files* section for the
full list.

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
