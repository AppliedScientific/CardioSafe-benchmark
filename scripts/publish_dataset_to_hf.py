"""Mirror the CardioSafe-benchmark dataset to HuggingFace Datasets.

Publishes to `appliedscientific/cardiosafe-benchmark` (repo_type=dataset).

Layout:
  - Pre-joined splits per (split_strategy × version), one Parquet shard per
    fold (train/val/test), surfaced as named configs so users can do
    `load_dataset("appliedscientific/cardiosafe-benchmark", "tan70_v1_1", split="test")`.
  - Raw canonical files (compounds, labels, splits) as separate Parquets
    so anyone doing custom merges has the source-of-truth.
  - Supplementary materials (tables S0–S9, Notes S1–S3, Figure S1)
    verbatim — same CSV/MD/JSON/PDF/PNG as the GitHub deposit.
  - Comparator predictions (CToxPred2, CardioGenAI) as Parquet.

The CardioSafe-benchmark GitHub repo remains the authoritative source.
HF Datasets is a mirror + ML-benchmark-loading convenience.

Usage:
  python scripts/publish_dataset_to_hf.py                # full publish
  python scripts/publish_dataset_to_hf.py --dry-run      # stage only
  python scripts/publish_dataset_to_hf.py --skip-build   # re-upload existing staging
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_STAGING = REPO_ROOT / "build" / "hf_dataset_staging"
REPO_ID = "appliedscientific/cardiosafe-benchmark"

SPLIT_STRATEGIES = ("tan70", "tan60")
VERSIONS = {
    "v1.0": {"tan70": "tan70.csv", "tan60": "tan60.csv"},
    "v1.1": {"tan70": "tan70_v1_1.csv", "tan60": "tan60_v1_1.csv"},
}
FOLD_NAMES = {"train": "train", "val": "validation", "test": "test"}


def load_hf_token() -> str:
    if (token := os.environ.get("HF_TOKEN")):
        return token
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN not found in environment or .env")


def _to_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy", engine="pyarrow")


def build_staging(staging: Path) -> None:
    import pandas as pd

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    print("Loading compounds, labels, splits...")
    compounds = pd.read_csv(DATA_ROOT / "compounds" / "compounds.csv")
    labels = pd.read_csv(DATA_ROOT / "labels" / "labels_v1.csv")
    splits_by_name = {
        name: pd.read_csv(DATA_ROOT / "splits" / fname)
        for name, fname in {
            "tan70": "tan70.csv",
            "tan60": "tan60.csv",
            "tan70_v1_1": "tan70_v1_1.csv",
            "tan60_v1_1": "tan60_v1_1.csv",
        }.items()
    }

    # ── raw canonical Parquets ───────────────────────────────────────────────
    print("Writing raw Parquets (compounds, labels, splits)...")
    _to_parquet(compounds, staging / "raw" / "compounds.parquet")
    _to_parquet(labels, staging / "raw" / "labels.parquet")
    for name, df in splits_by_name.items():
        _to_parquet(df, staging / "raw" / "splits" / f"{name}.parquet")

    # ── pre-joined per-fold Parquets per (split × version) ───────────────────
    label_cols = [
        "herg_pchembl",
        "herg_blocker_10um",
        "herg_blocker_1um",
        "nav15_pchembl",
        "nav15_blocker",
        "cav12_pchembl",
        "cav12_blocker",
        "iks_blocker",
    ]
    merged_labels = compounds.merge(
        labels[["row_idx"] + label_cols], on="row_idx", how="left"
    )

    for version, split_files in VERSIONS.items():
        for strategy, split_csv in split_files.items():
            split_df = pd.read_csv(DATA_ROOT / "splits" / split_csv)
            joined = merged_labels.merge(
                split_df[["row_idx", "fold"]], on="row_idx", how="inner"
            )
            config_name = f"{strategy}_{version.replace('.', '_')}"
            out_dir = staging / config_name
            for fold_src, fold_dst in FOLD_NAMES.items():
                shard = joined.query("fold == @fold_src").drop(columns=["fold"])
                _to_parquet(shard, out_dir / f"{fold_dst}.parquet")
            print(f"  {config_name}: {len(joined):,} rows total")

    # ── comparators as Parquet ───────────────────────────────────────────────
    print("Converting comparator predictions...")
    for name in ("ctoxpred2_tan70_predictions", "cardiogenai_tan70_predictions"):
        df = pd.read_csv(DATA_ROOT / "comparators" / f"{name}.csv")
        _to_parquet(df, staging / "comparators" / f"{name}.parquet")
    # Keep comparator README verbatim
    shutil.copy(
        DATA_ROOT / "comparators" / "README.md",
        staging / "comparators" / "README.md",
    )

    # ── supplementary materials verbatim ─────────────────────────────────────
    print("Copying supplementary materials...")
    supp_dst = staging / "supplementary"
    supp_dst.mkdir(parents=True, exist_ok=True)
    for f in (DATA_ROOT / "supplementary").iterdir():
        if f.is_file():
            shutil.copy(f, supp_dst / f.name)

    # ── data/README.md and labels MANIFEST verbatim ──────────────────────────
    shutil.copy(DATA_ROOT / "README.md", staging / "raw" / "README.md")
    shutil.copy(DATA_ROOT / "labels" / "MANIFEST.json", staging / "raw" / "labels_MANIFEST.json")
    shutil.copy(DATA_ROOT / "splits" / "CHANGELOG.md", staging / "raw" / "splits_CHANGELOG.md")

    write_dataset_card(staging)


def write_dataset_card(staging: Path) -> None:
    """HF dataset card: YAML frontmatter (license, configs, tags) + body."""
    configs_yaml = "\n".join(
        f"""  - config_name: {strategy}_{ver.replace('.', '_')}
    data_files:
      - split: train
        path: {strategy}_{ver.replace('.', '_')}/train.parquet
      - split: validation
        path: {strategy}_{ver.replace('.', '_')}/validation.parquet
      - split: test
        path: {strategy}_{ver.replace('.', '_')}/test.parquet"""
        for ver in VERSIONS
        for strategy in SPLIT_STRATEGIES
    )

    card = f"""---
license: cc-by-4.0
language:
  - en
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - chemistry
  - drug-discovery
  - cardiotoxicity
  - hERG
  - Nav1.5
  - Cav1.2
  - IKs
  - ion-channels
  - QSAR
  - ChEMBL
  - cipa
  - benchmark
pretty_name: CardioSafe ion-channel benchmark
size_categories:
  - 100K<n<1M
configs:
{configs_yaml}
---

# CardioSafe ion-channel benchmark

Curated multi-task ion-channel labels + Tanimoto-controlled splits +
supplementary materials for **CardioSafe: multi-task prediction of
cardiac ion channel activity with reverse-leak audited benchmarking**
(Jovanović et al., 2026,
[bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1)).

The canonical source is the
[CardioSafe-benchmark GitHub repository](https://github.com/AppliedScientific/CardioSafe-benchmark).
This HF Datasets repo mirrors `data/` from that deposit and adds
pre-joined per-fold Parquet shards so you can write:

```python
from datasets import load_dataset

# v1.1 (audit-clean) test fold on the tan70 Tanimoto cutoff:
ds = load_dataset("appliedscientific/cardiosafe-benchmark",
                  "tan70_v1_1", split="test")
print(ds.column_names)
# ['row_idx', 'smiles', 'inchikey', 'herg_pchembl', 'herg_blocker_10um',
#  'herg_blocker_1um', 'nav15_pchembl', 'nav15_blocker', 'cav12_pchembl',
#  'cav12_blocker', 'iks_blocker']
```

Each row is one curated compound. Label columns are NaN-sparse — only
those compounds with primary-screen evidence for the relevant channel
have non-null values.

## Configs (pre-joined splits)

| Config | Split strategy | Version | Compounds (train / val / test) |
| --- | --- | --- | --- |
| `tan70_v1_0` | Tanimoto ≥ 0.70 cutoff | v1.0 preprint | 241,792 / 46,326 / 46,326 |
| `tan60_v1_0` | Tanimoto ≥ 0.60 cutoff | v1.0 preprint | 306,665 / 13,889 / 13,889 |
| `tan70_v1_1` | Tanimoto ≥ 0.70 cutoff | v1.1 retrain | 241,790 / 46,328 / 46,326 |
| `tan60_v1_1` | Tanimoto ≥ 0.60 cutoff | v1.1 retrain | 306,662 / 13,892 / 13,889 |

**v1.1 differs from v1.0 by 2 force-routed analogs in the cardiac-cliff
cluster** (terfenadine/fexofenadine/HMT). The test fold is identical
across v1.0 and v1.1 — paper Table 2/3 metrics are unchanged. See
[`supplementary/note_s3_v1_1_audit_correction.md`](https://github.com/AppliedScientific/CardioSafe-benchmark/blob/main/data/supplementary/note_s3_v1_1_audit_correction.md).

## Label schema

| Column | Head | Type |
| --- | --- | --- |
| `herg_pchembl` | regression — hERG pIC50 | float |
| `herg_blocker_10um` | hERG blocker @ 10 µM | binary {{0, 1}} |
| `herg_blocker_1um` | hERG blocker @ 1 µM | binary {{0, 1}} |
| `nav15_pchembl` | regression — Nav1.5 pIC50 | float |
| `nav15_blocker` | Nav1.5 blocker @ 10 µM | binary {{0, 1}} |
| `cav12_pchembl` | regression — Cav1.2 pIC50 | float |
| `cav12_blocker` | Cav1.2 blocker @ 10 µM | binary {{0, 1}} |
| `iks_blocker` | IKs blocker @ 10 µM (exploratory) | binary {{0, 1}} |

IKs has no regression head (n = 115 labelled compounds; treated as
exploratory).

Per-channel label counts (primary binary head, all folds combined):

| Channel | n labelled | n blockers |
| --- | --- | --- |
| hERG (10 µM) | 331,127 | 11,881 |
| Nav1.5 (10 µM) | 3,160 | 1,240 |
| Cav1.2 (10 µM) | 1,138 | 548 |
| IKs (10 µM) | 115 | 30 |

## Raw canonical files

`raw/` contains the source-of-truth files exactly as published in the
GitHub repo, for power users doing custom merges:

```
raw/
├── compounds.parquet              # 334,444 × (row_idx, smiles, inchikey)
├── labels.parquet                 # 334,444 × 8 sparse labels
├── splits/
│   ├── tan70.parquet              # v1.0 — paper preprint splits
│   ├── tan60.parquet
│   ├── tan70_v1_1.parquet         # v1.1 — audit-clean retrain splits
│   └── tan60_v1_1.parquet
├── labels_MANIFEST.json           # curation provenance (sources, voting policy)
├── splits_CHANGELOG.md            # v1.0 → v1.1 diff
└── README.md                      # full data-deposit notes from GitHub
```

All splits share the same `row_idx` keying — join on it for arbitrary
slicing.

## Comparators

`comparators/` ships the CToxPred2 and CardioGenAI predictions on the
v1.0 `tan70` test fold, the inputs to the reverse-leak audit and the
head-to-head comparison in paper Tables 3 / 3b / S2 / S3 / Figure 4.

## Supplementary materials

`supplementary/` ships verbatim:
- **Notes S1, S2, S3** — Y-randomization mechanism; activity-cliff
  curation provenance + bibliography + per-pair composition + filtered
  cliff manifests; audit-driven v1.1 correction + retrain metrics + the
  cardiac-cliff case study
- **Tables S0, S1, S2, S3, S5, S6, S7, S8, S9** — descriptor spec,
  per-head confusion matrices, comparator panels pre/post de-leak, tan60
  drug panel, failure-mode SMILES, AD per-bin metrics, L1000 threshold
  sweep, curation sensitivity (no S4 — paper supplementary numbering
  jumps S3 → S5)
- **Figure S1** — hERG reliability curves across applicability-domain
  bins (PDF + PNG + JSON of the underlying decile data)

## Source data

- **Labels** are derived from ChEMBL 36 (source dump SHA-256
  `b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`,
  available from <https://www.ebi.ac.uk/chembl/>) and the **hERG Central
  primary screen**, under a pharmacology-aware curation policy that
  retains censored values and inhibition-percentage votes. Full
  provenance lives in `raw/labels_MANIFEST.json`.
- **Splits** are Tanimoto-controlled across train / val / test on
  Morgan-r2-2048 fingerprints, with terfenadine and fexofenadine
  force-routed to `val` so the canonical cardiac activity cliff is
  available as a held-out case study. v1.1 additionally force-routes the
  hydroxymethyl-terfenadine analogs flagged by an exhaustive
  O(n_train × n_other) Tanimoto leakage audit
  (`scripts/audit_tanimoto_leak.py` in the GitHub repo).

## What is **not** here

- **L1000 raw signatures** used to train the expression encoder —
  available from GEO under GSE92742 (Phase I) and GSE70138 (Phase II).
- **Continually-updated production ensemble** — served at
  [platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe).
  This deposit is the frozen paper snapshot.
- **Model weights** — at
  [`appliedscientific/cardiosafe`](https://huggingface.co/appliedscientific/cardiosafe)
  (5-seed v1.0 + v1.1 ensembles + L1000 encoder, CC-BY-NC-4.0).
- **Interactive demo** — at
  [`appliedscientific/cardiosafe` (Space)](https://huggingface.co/spaces/appliedscientific/cardiosafe).

## License

CC-BY-4.0. Use with attribution; commercial use allowed under the
license terms. See the
[full LICENSE-DATA](https://github.com/AppliedScientific/CardioSafe-benchmark/blob/main/LICENSE-DATA)
in the GitHub repo for the exact text.

Note: the *model weights* are CC-BY-NC-4.0 (non-commercial), not the
data. They live at a separate HF repo —
[`appliedscientific/cardiosafe`](https://huggingface.co/appliedscientific/cardiosafe).

## Citation

```bibtex
@article{{cardiosafe2026,
  title   = {{CardioSafe: multi-task prediction of cardiac ion channel
             activity with reverse-leak audited benchmarking}},
  author  = {{Jovanović, Mihailo and Weidener, Lukas and Brkić, Marko and
             Ulgac, Emre and Meduri, Aakaash}},
  year    = {{2026}},
  journal = {{bioRxiv}},
  doi     = {{10.64898/2026.05.06.723181}},
  url     = {{https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1}}
}}
```
"""
    (staging / "README.md").write_text(card)


def upload(staging: Path, token: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Ensuring dataset repo {REPO_ID} (private={private})")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )
    print(f"Uploading {staging} -> {REPO_ID}")
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(staging),
        repo_type="dataset",
        commit_message="Publish CardioSafe-benchmark dataset (compounds, labels, splits, supplementary)",
    )
    print(f"Done. https://huggingface.co/datasets/{REPO_ID}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    p.add_argument("--private", action="store_true")
    p.add_argument("--skip-build", action="store_true",
                   help="Reuse existing staging directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build staging only, do not upload.")
    args = p.parse_args()

    if args.skip_build:
        if not args.staging.exists():
            sys.exit(f"--skip-build set but staging does not exist: {args.staging}")
        print(f"Reusing staging at {args.staging}")
    else:
        build_staging(args.staging)

    n_files = sum(1 for f in args.staging.rglob("*") if f.is_file())
    total_mb = sum(f.stat().st_size for f in args.staging.rglob("*") if f.is_file()) / 1e6
    print(f"Staging ready: {n_files} files, {total_mb:.1f} MB")

    if args.dry_run:
        print("--dry-run set, skipping upload.")
        return 0

    token = load_hf_token()
    upload(args.staging, token, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
