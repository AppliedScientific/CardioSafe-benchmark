# Data

All files in this tree are released under
[CC BY 4.0](../LICENSE-DATA). The benchmark's row order is canonical:
the integer in the `row_idx` column points at the same compound across every
file in this tree (and at the matching row of `labels/labels.npy` once that
is released).

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

## Loading example

```python
import pandas as pd

compounds = pd.read_csv("data/compounds/compounds.csv")
tan70     = pd.read_csv("data/splits/tan70.csv")

test_smiles = (
    tan70.query("fold == 'test'")
         .merge(compounds[["row_idx", "smiles"]], on="row_idx")
)
print(len(test_smiles))   # 46326
```
