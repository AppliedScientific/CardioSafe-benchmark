# CardioSafe-benchmark

[![License: MIT + CC-BY-4.0 + CC-BY-NC-4.0](https://img.shields.io/badge/license-MIT_%2B_CC--BY--4.0_%2B_CC--BY--NC--4.0-blue.svg)](LICENSE)
[![Preprint: bioRxiv](https://img.shields.io/badge/preprint-bioRxiv-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1)
[![Inference: ASI Platform](https://img.shields.io/badge/inference-platform.appliedscientific.ai-2ea44f.svg)](https://platform.appliedscientific.ai/cardiosafe)
[![Weights: GitHub Releases](https://img.shields.io/badge/weights-GitHub_Releases-181717.svg)](https://github.com/AppliedScientific/CardioSafe-benchmark/releases)
[![Applied Scientific Intelligence](https://img.shields.io/badge/lab-appliedscientific.ai-000000.svg)](https://appliedscientific.ai)

Splits, labels, and supplementary artifacts for **CardioSafe: multi-task
prediction of cardiac ion channel activity with reverse-leak audited
benchmarking** (Jovanović et al., 2026).

CardioSafe is a three-branch multi-task neural network that predicts blocker
status and pIC50 for the four CiPA cardiac ion channels — hERG, Nav1.5,
Cav1.2, and (exploratory) IKs — trained on the largest publicly reported
multi-channel cardiac ion channel dataset (ChEMBL 36 + hERG Central). This
repository is the public deposit for the paper: the curated label
matrix, Tanimoto-controlled splits, supplementary tables / notes / figure
promised in the *Additional files* section of the manuscript, the
exhaustive Tanimoto leakage audit, the reference model + training-step
code, and the v1.0 / v1.1 ensemble weights with runnable inference.

> **Status:** initial release. The repository ships the curated dataset
> + Tanimoto-controlled splits + supplementary tables / notes / figure,
> together with the paper-faithful reference architecture, training-step
> reference code, the exhaustive O(n²) Tanimoto leakage audit script
> that drove the v1.1 correction, **and the v1.0 + v1.1 ensemble weights
> + runnable inference** (`inference/predict.py`). The deployed CardioSafe
> ensemble at
> [platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe)
> is continually updated with CRO-validated bioassay data and may differ
> from this paper snapshot. Questions: lukas@appliedscientific.ai or
> mihailo@appliedscientific.ai.

## What's in the box

| Path | Contents |
| --- | --- |
| `data/compounds/compounds.csv` | Canonical SMILES + standard InChI-keys for all 334,444 curated compounds |
| `data/labels/labels_v1.csv` + `MANIFEST.json` | The 8-task curated label matrix and its provenance manifest |
| `data/splits/tan70.csv`, `tan60.csv` | v1.0 splits — Tanimoto ≥ 0.70 / 0.60 cutoffs, as used in the bioRxiv preprint |
| `data/splits/tan70_v1_1.csv`, `tan60_v1_1.csv` | **v1.1** splits — audit-clean (cardiac-cliff cluster fully force-routed to val). Test fold identical to v1.0. See `data/splits/CHANGELOG.md`. |
| `data/supplementary/` | Notes S1–S3, Tables S0–S9 (no S4), Figure S1, plus the per-split cliff manifests |
| `data/comparators/` | CToxPred2 and CardioGenAI predictions on the tan70 test fold — the inputs to the reverse-leak audit and the head-to-head comparison in Tables 3 / 3b / S2 / S3 / Figure 4 |
| `scripts/audit_tanimoto_leak.py` | Exhaustive O(n_train × n_other) Tanimoto leakage audit — verifies no cross-fold Morgan-r2-2048 edges at or above the cutoff. See [`scripts/README.md`](scripts/README.md). |
| `model/` | Reference architecture (`CrossAttnIonChannelPredictor`, ChemBERTa adapter, L1000 encoder) + `MODEL_CARD.md`. |
| `inference/` | Runnable inference: `predict.py` (CSV-in / CSV-out CLI), `featurize.py`, `ensemble.py`. Auto-downloads weights from [Releases](https://github.com/AppliedScientific/CardioSafe-benchmark/releases) on first call. See [`inference/README.md`](inference/README.md). |
| `train/` | Paper-faithful loss functions and Stage 1 / Stage 2 training-step references. No data loaders; bring your own feature caches. |
| [Releases](https://github.com/AppliedScientific/CardioSafe-benchmark/releases) | v1.0 + v1.1 ensemble weights (5 seeds each, 15 MB / seed) and the L1000 expression encoder (10 MB). Licensed CC-BY-NC-4.0 (see `LICENSE-WEIGHTS`). |

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

Two Tanimoto-controlled splits are released in two versions:

| version | split | train | val | test | excluded |
| --- | --- | --- | --- | --- | --- |
| v1.0 (preprint) | tan70 (primary)  | 241,792 | 46,326 | 46,326 | 0 |
| v1.0 (preprint) | tan60 (stricter) | 306,665 | 13,889 | 13,889 | 1 |
| **v1.1** (audit-clean) | tan70 | 241,790 | 46,328 | 46,326 | 0 |
| **v1.1** (audit-clean) | tan60 | 306,662 | 13,892 | 13,889 | 1 |

Both versions force-route terfenadine and fexofenadine to `val` so the
canonical cardiac activity cliff is available as a held-out case study.
v1.1 additionally force-routes 2 hydroxymethyl-terfenadine analogs on
tan70 and 3 on tan60 in the same cluster (caught by
`scripts/audit_tanimoto_leak.py`); test fold is identical between
versions. Use v1.1 for new work; v1.0 is retained
so the bioRxiv preprint numbers stay reproducible. See
[`data/splits/CHANGELOG.md`](data/splits/CHANGELOG.md) and Note S3
([`data/supplementary/note_s3_v1_1_audit_correction.md`](data/supplementary/note_s3_v1_1_audit_correction.md)).

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

- **Notes S1, S2, S3** — Y-randomization mechanism; activity-cliff
  curation provenance + 25-source bibliography + per-`pair_id`
  composition + per-split filtered cliff manifests; audit-driven v1.1
  correction with retrain metrics and terfenadine / fexofenadine
  case-study re-evaluation
- **Tables S0, S1, S2, S3, S5, S6, S7, S8, S9** — descriptor spec,
  per-head confusion matrices, comparator panels pre/post de-leak, tan60
  drug panel, failure-mode SMILES, AD per-bin metrics, L1000 threshold
  sweep, curation sensitivity (no S4 — the supplementary numbering
  jumps S3 → S5 in the manuscript)
- **Figure S1** — hERG reliability curves across applicability-domain
  bins (PDF + PNG + JSON of the underlying decile data)

## What's NOT in this repository

- **L1000 raw signatures** used to train the expression encoder are
  available from GEO under accessions GSE92742 (Phase I) and GSE70138
  (Phase II). The trained encoder weights are released; the raw input
  signatures are not redistributed.
- **ChEMBL 36 source dump** (SHA-256
  `b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`) is
  available from <https://www.ebi.ac.uk/chembl/>.
- **Continually-updated production ensemble** is served at
  [platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe)
  and may differ from this paper snapshot — it is retrained on new
  CRO-validated bioassay data routed through ASI's partner network.
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

- **Code** in this repository is under the [MIT License](LICENSE).
- **Data** (everything under `data/`) is released under
  [Creative Commons Attribution 4.0 International](LICENSE-DATA).
- **Model weights** distributed via the GitHub Releases of this
  repository are released under
  [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE-WEIGHTS).
  Academic, educational, and non-profit research use is permitted with
  attribution. Commercial use requires a separate license — contact the
  authors (`lukas@appliedscientific.ai`, `mihailo@appliedscientific.ai`).
