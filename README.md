# CardioSafe-benchmark

Splits, labels, and supplementary artifacts for **CardioSafe: multi-task
prediction of cardiac ion channel activity with reverse-leak audited
benchmarking** (Jovanović et al., 2026).

📄 **Preprint:** [biorxiv.org/content/10.64898/2026.05.06.723181v1](https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1)

CardioSafe is a three-branch multi-task neural network that predicts blocker
status and pIC50 for the four CiPA cardiac ion channels — hERG, Nav1.5,
Cav1.2, and (exploratory) IKs — trained on the largest publicly reported
multi-channel cardiac ion channel dataset (ChEMBL 36 + hERG Central). This
repository hosts the **data deposit** for the paper: the curated label
matrix, Tanimoto-controlled splits, and supplementary tables / notes /
figure promised in the *Additional files* section of the manuscript.

> **Status:** initial release. This repository is a **data deposit**.
> Inference is available via a free academic API at the journal version
> of the manuscript; researchers seeking earlier access for benchmarking
> studies can contact the corresponding author
> (Lukas@appliedscientific.ai).

## What's in the box

| Path | Contents |
| --- | --- |
| `data/compounds/compounds.csv` | Canonical SMILES + standard InChI-keys for all 334,444 curated compounds |
| `data/labels/labels_v1.csv` + `MANIFEST.json` | The 8-task curated label matrix and its provenance manifest |
| `data/splits/tan70.csv` | tan70 split indices (Tanimoto ≥ 0.70 cutoff) — primary benchmark |
| `data/splits/tan60.csv` | tan60 split indices (Tanimoto ≥ 0.60 cutoff) — stricter secondary benchmark |
| `data/supplementary/` | Notes S1–S2, Tables S0–S9 (no S4), Figure S1, plus the per-split cliff manifests |

See [`data/README.md`](data/README.md) for the schema, fold counts, and per-file documentation.

## Quickstart

```python
import pandas as pd

compounds = pd.read_csv("data/compounds/compounds.csv")
tan70     = pd.read_csv("data/splits/tan70.csv")
labels    = pd.read_csv("data/labels/labels_v1.csv")

# hERG 10 µM test fold (n ≈ 46,120, matching paper Table 2)
test_herg_10um = (
    labels[["row_idx", "herg_blocker_10um"]]
        .dropna()
        .merge(tan70.query("fold == 'test'"), on="row_idx")
        .merge(compounds[["row_idx", "smiles"]], on="row_idx")
)
print(len(test_herg_10um))
```

## Splits

Two Tanimoto-controlled splits are released, built off Morgan radius-2
2048-bit Tanimoto with cross-fold edges forbidden:

| split | train | val | test | excluded |
| --- | --- | --- | --- | --- |
| tan70 (primary)  | 241,792 | 46,326 | 46,326 | 0 |
| tan60 (stricter) | 306,665 | 13,889 | 13,889 | 1 |

Both splits force-exclude terfenadine and fexofenadine from the training
fold (routed to `val`) so the canonical cardiac activity cliff is
available as a held-out case study.

## Labels

`labels_v1.csv` is a NaN-sparse `(334,444 × 8)` matrix:

| col | head | type |
| --- | --- | --- |
| `herg_pchembl` | regression | hERG pIC50 |
| `herg_blocker_10um` | binary | hERG blocker @ 10 µM |
| `herg_blocker_1um` | binary | hERG blocker @ 1 µM |
| `nav15_pchembl` | regression | Nav1.5 pIC50 |
| `nav15_blocker` | binary | Nav1.5 blocker @ 10 µM |
| `cav12_pchembl` | regression | Cav1.2 pIC50 |
| `cav12_blocker` | binary | Cav1.2 blocker @ 10 µM |
| `iks_blocker` | binary | IKs blocker @ 10 µM (exploratory) |

Per-channel counts (primary binary head):

| channel | n labelled | n blockers |
| --- | --- | --- |
| hERG    (10 µM) | 331,127 | 11,881 |
| Nav1.5  (10 µM) | 3,160   | 1,240  |
| Cav1.2  (10 µM) | 1,138   | 548    |
| IKs     (10 µM) | 115     | 30     |

Labels are derived from ChEMBL 36 (source dump SHA-256
`b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`) and
the hERG Central primary screen, under a pharmacology-aware curation
policy that retains censored values and inhibition-percentage votes.
Full provenance is in [`data/labels/MANIFEST.json`](data/labels/MANIFEST.json).

## Supplementary materials

Everything the manuscript's *Additional files* section promises is in
[`data/supplementary/`](data/supplementary/), with three formats per
table (CSV / Markdown / JSON) where applicable:

- **Notes S1, S2** — Y-randomization mechanism, activity-cliff curation
  provenance + 25-source bibliography + per-`pair_id` composition +
  per-split filtered cliff manifests
- **Tables S0, S1, S2, S3, S5, S6, S7, S8, S9** — descriptor spec,
  per-head confusion matrices, comparator panels pre/post de-leak, tan60
  drug panel, failure-mode SMILES, AD per-bin metrics, L1000 threshold
  sweep, curation sensitivity (no S4 — the supplementary numbering
  jumps S3 → S5 in the manuscript)
- **Figure S1** — hERG reliability curves across applicability-domain
  bins (PDF + PNG + JSON of the underlying decile data)

## What's NOT in this repository

The paper's *Availability of data and materials* section deposits only
data; inference and code are available elsewhere.

- **Model weights** are not deposited here. Inference is via a free
  academic API at the journal version of the manuscript; contact the
  corresponding author for earlier access.
- **L1000 raw signatures** used to train the expression encoder are
  available from GEO under accessions GSE92742 (Phase I) and GSE70138
  (Phase II).
- **ChEMBL 36 source dump** (SHA-256
  `b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`) is
  available from <https://www.ebi.ac.uk/chembl/>.
- **Comparator predictions** (CToxPred2, CardioGenAI) used in the
  reverse-leak audit are reproducible from each comparator's released
  code on this repo's `tan70` test fold. The aggregated paired-bootstrap
  comparison numbers are shipped in Tables S2 / S3.

## Citation

```bibtex
@article{cardiosafe2026,
  title   = {CardioSafe: multi-task prediction of cardiac ion channel
             activity with reverse-leak audited benchmarking},
  author  = {Jovanović, Mihailo and Weidener, Lukas and Brkić, Marko and
             Ulgac, Emre and Meduri, Aakaash},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.64898/2026.05.06.723181},
  url     = {https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1}
}
```

## License

- **Code** snippets in this repository (and any future code releases)
  are under the [MIT License](LICENSE).
- **Data** (everything under `data/`) is released under
  [Creative Commons Attribution 4.0 International](LICENSE-DATA).
