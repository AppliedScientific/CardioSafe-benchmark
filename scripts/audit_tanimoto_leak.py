"""
Exhaustive O(n_train x n_other) Tanimoto leakage audit for the tan70 / tan60
splits in this repository.

For each train compound, computes the Morgan radius-2 2048-bit Tanimoto
similarity against every val and every test compound, and reports any pair
that meets or exceeds the cutoff. The splits were built with a graph
constraint that forbids cross-fold edges at the cutoff, so a clean deposit
should produce zero violations.

Reads only files from data/ in this repository. Writes a CSV listing
violations (or no violations) to the path given by --out.

Usage:
    # full tan70 audit (cutoff 0.70, ~22 billion comparisons)
    python scripts/audit_tanimoto_leak.py --split tan70

    # full tan60 audit (cutoff 0.60)
    python scripts/audit_tanimoto_leak.py --split tan60

    # smoke test on a 200-compound train sample (seconds, not minutes)
    python scripts/audit_tanimoto_leak.py --split tan70 --sample 200

Dependencies:
    rdkit >= 2025.9.6
    numpy >= 2.0

Runtime on a modern laptop:
    tan70 full:   ~15-45 min (241,792 train x 92,652 val+test)
    tan60 full:   ~6-20 min  (306,665 train x 27,778 val+test)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--split",
        default="tan70",
        help="Split name (file at data/splits/<split>.csv). Default tan70. "
        "Cutoff is inferred from the prefix: any name starting with 'tan70' "
        "uses 0.70; any name starting with 'tan60' uses 0.60. Override with --cutoff.",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Override the Tanimoto cutoff (default: 0.70 if --split starts with tan70, 0.60 if tan60).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the data/ directory (default: <repo>/data).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV for violations (default: audit_<split>_leaks.csv next to the script).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: randomly subsample this many train compounds (smoke test). Default: full audit.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260513,
        help="Random seed for --sample.",
    )
    return p.parse_args()


def load_split(split_path: Path) -> dict[int, str]:
    fold_for_row: dict[int, str] = {}
    with split_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fold_for_row[int(row["row_idx"])] = row["fold"]
    return fold_for_row


def load_compounds(compounds_path: Path) -> tuple[list[int], list[str]]:
    row_idxs: list[int] = []
    smiles: list[str] = []
    with compounds_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            row_idxs.append(int(row["row_idx"]))
            smiles.append(row["smiles"])
    return row_idxs, smiles


def fingerprint_all(smiles: list[str]) -> list:
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    skipped = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(None)
            skipped += 1
        else:
            out.append(fpgen.GetFingerprint(mol))
    if skipped:
        print(f"  WARNING: {skipped} SMILES failed to parse; those rows are excluded", file=sys.stderr)
    return out


def main() -> int:
    args = parse_args()

    if args.cutoff is not None:
        cutoff = args.cutoff
    elif args.split.startswith("tan70"):
        cutoff = 0.70
    elif args.split.startswith("tan60"):
        cutoff = 0.60
    else:
        sys.exit(f"cannot infer cutoff for --split={args.split!r}; pass --cutoff explicitly")
    out_path = args.out or (Path.cwd() / f"audit_{args.split}_leaks.csv")

    compounds_path = args.data_dir / "compounds" / "compounds.csv"
    split_path = args.data_dir / "splits" / f"{args.split}.csv"
    for p in (compounds_path, split_path):
        if not p.exists():
            sys.exit(f"missing required input: {p}")

    print(f"Loading {args.split} split + compounds ...", flush=True)
    fold_for_row = load_split(split_path)
    row_idxs, smiles = load_compounds(compounds_path)
    n = len(row_idxs)
    if n != len(fold_for_row):
        sys.exit(f"compounds.csv has {n} rows but {split_path.name} has {len(fold_for_row)}")

    print("Computing Morgan-r2-2048 fingerprints (parses all SMILES once) ...", flush=True)
    t0 = time.time()
    fps = fingerprint_all(smiles)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    fold_arr = np.array([fold_for_row[r] for r in row_idxs])
    train_pos = np.where(fold_arr == "train")[0]
    other_pos = np.where((fold_arr == "val") | (fold_arr == "test"))[0]

    train_pos = np.array([i for i in train_pos if fps[i] is not None], dtype=np.int64)
    other_pos = np.array([i for i in other_pos if fps[i] is not None], dtype=np.int64)

    if args.sample is not None and args.sample < len(train_pos):
        rng = random.Random(args.seed)
        chosen = rng.sample(list(train_pos), args.sample)
        train_pos = np.array(sorted(chosen), dtype=np.int64)
        print(f"  (sampling {args.sample:,} of train fold for smoke test)", flush=True)

    n_train = len(train_pos)
    n_other = len(other_pos)
    print(f"  train: {n_train:,}    val+test: {n_other:,}", flush=True)
    print(f"  comparisons: {n_train * n_other:,}", flush=True)
    print(f"  cutoff: Tanimoto >= {cutoff:.3f}", flush=True)

    other_fps = [fps[i] for i in other_pos]

    t0 = time.time()
    n_leak = 0
    report_every = max(1, n_train // 50)

    with out_path.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["row_idx_train", "row_idx_other", "fold_other", "tanimoto"])

        for k, ti in enumerate(train_pos, start=1):
            sims = DataStructs.BulkTanimotoSimilarity(fps[ti], other_fps)
            sims_arr = np.asarray(sims)
            hits = np.where(sims_arr >= cutoff)[0]
            if len(hits):
                row_train = row_idxs[ti]
                for h in hits:
                    oi = other_pos[h]
                    w.writerow([row_train, row_idxs[oi], fold_for_row[row_idxs[oi]], f"{sims_arr[h]:.4f}"])
                n_leak += len(hits)

            if k % report_every == 0 or k == n_train:
                elapsed = time.time() - t0
                rate = k / max(elapsed, 1e-9)
                eta = (n_train - k) / max(rate, 1e-9)
                print(
                    f"  {k:>8,}/{n_train:,} train  leaks={n_leak:,}  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                    flush=True,
                )

    print("=" * 60)
    print(f"AUDIT COMPLETE  split={args.split}  cutoff={cutoff:.3f}")
    print(f"  cross-fold violations (train -> val/test): {n_leak:,}")
    if n_leak == 0:
        print(f"  PASS - no cross-fold edges at or above the cutoff.")
        out_path.unlink()
    else:
        print(f"  FAIL - violations written to {out_path}")
    print("=" * 60)
    return 0 if n_leak == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
