# CardioSafe-benchmark

Splits, labels, and supplementary artifacts for **CardioSafe: multi-task
prediction of cardiac ion channel activity with reverse-leak audited
benchmarking** (Jovanović et al., 2026).

CardioSafe is a three-branch multi-task neural network that predicts blocker
status and pIC50 for the four CiPA cardiac ion channels — hERG, Nav1.5,
Cav1.2, and (exploratory) IKs — trained on the largest publicly reported
multi-channel cardiac ion channel dataset (ChEMBL 36 + hERG Central). This
repository hosts the curated dataset and Tanimoto-controlled splits used in
the paper, plus the supplementary materials needed to reproduce the reported
numbers.

> **Status:** initial release. Inference is available via a free academic API
> at the journal version of the manuscript; researchers seeking earlier
> access for benchmarking studies can contact the corresponding author
> (Lukas@appliedscientific.ai).

## What's in the box

| Path | Contents |
| --- | --- |
| `data/splits/` | tan70 and tan60 split indices (CSV) — the primary benchmark splits |
| `data/labels/` | labels_v1 — 8-task curated label matrix keyed by InChI-key |
| `data/compounds/` | InChI-keys and canonical SMILES for all 334,444 curated compounds |
| `data/supplementary/` | Notes S1–S2, Tables S0–S9, Figure S1 |
| `data/leakage/` | Reverse-leak audit outputs (Tanimoto ≥ 0.70 and ≥ 0.99 neighbour masks) |
| `data/cliffs/` | Activity-cliff fine-tune curation (25-source manifest + per-split filtered sets) |
| `data/comparators/` | Comparator predictions on the v2 test fold (CToxPred2, CardioGenAI) |
| `scripts/` | Reproducibility scripts — split builders, training driver, evaluation, figure generation |

Large artifacts (model weights, full feature caches) are hosted on Zenodo;
see `data/ZENODO.md` for DOIs and SHA-256 checksums.

## Quickstart

```bash
# Install
uv pip install -e .

# Load the primary tan70 split
python -c "
import pandas as pd
df = pd.read_csv('data/splits/tan70.csv')
print(df.groupby('fold').size())
print(df.head())
"
```

## Splits

Two Tanimoto-controlled splits are released:

- **tan70 (primary)** — Morgan radius-2 2048-bit Tanimoto ≥ 0.70 forbids
  cross-bucket edges; 80/10/10 train/val/test. Terfenadine and fexofenadine
  are force-excluded from the training fold.
- **tan60 (stricter)** — same construction at Tanimoto ≥ 0.60.

Both splits are global (not per-head). Each row in the CSV is one compound;
columns are `row_idx`, `inchikey`, `smiles`, and `fold ∈ {train, val,
test}`.

## Labels

The `labels_v1` dataset is a NaN-sparse `(334,444 × 8)` matrix:

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

Labels are derived from ChEMBL 36 (SHA-256
`b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`) and the
hERG Central primary screen, under a pharmacology-aware curation policy that
retains censored values and inhibition-percentage votes. Full provenance is
in `data/labels/MANIFEST.json`.

## Reproducing the paper

See `docs/REPRODUCING.md` for the end-to-end walkthrough from raw ChEMBL to
the figures in the manuscript.

## Citation

```bibtex
@article{cardiosafe2026,
  title   = {CardioSafe: multi-task prediction of cardiac ion channel
             activity with reverse-leak audited benchmarking},
  author  = {Jovanović, Mihailo and Weidener, Lukas and Brkić, Marko and
             Ulgac, Emre and Meduri, Aakaash},
  year    = {2026},
  journal = {(submitted)},
  url     = {https://github.com/AppliedScientific/CardioSafe-benchmark}
}
```

## License

- **Code** is released under the [MIT License](LICENSE).
- **Data** (under `data/` and Zenodo-hosted artifacts) is released under
  [Creative Commons Attribution 4.0 International](LICENSE-DATA).
