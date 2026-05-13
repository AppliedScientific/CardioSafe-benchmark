# Scripts

Self-contained tools that operate on the public artifacts in `data/`.
None of these scripts need anything outside this repository.

## `audit_tanimoto_leak.py` — exhaustive cross-fold Tanimoto audit

Verifies the tan70 / tan60 splits in `data/splits/` do not contain any
cross-fold Morgan radius-2 2048-bit Tanimoto edges at or above the
declared cutoff. The splits were built with a graph constraint that
forbids such edges, so a clean deposit produces zero violations.

```bash
# full tan70 audit (cutoff 0.70, ~22B comparisons, ~15-45 min)
python scripts/audit_tanimoto_leak.py --split tan70

# full tan60 audit (cutoff 0.60, ~8B comparisons, ~6-20 min)
python scripts/audit_tanimoto_leak.py --split tan60

# fast smoke test on 200 train compounds
python scripts/audit_tanimoto_leak.py --split tan70 --sample 200
```

Output:

- Prints a per-`~2%`-progress line showing leaks-so-far and ETA.
- Writes violations (if any) to `audit_<split>_leaks.csv` next to the
  CWD, with columns `row_idx_train, row_idx_other, fold_other, tanimoto`.
- Exits with status `0` on PASS (zero violations) or `1` on FAIL.
- Deletes the empty violations CSV on PASS to keep the directory clean.

Algorithm: O(n_train x n_other) exhaustive — for every train compound,
compute Tanimoto against every val and every test compound via RDKit's
`DataStructs.BulkTanimotoSimilarity`. No locality-sensitive hashing or
nearest-neighbor index — the point is to make the audit obvious to read,
not fast at the asymptotic level.

Dependencies (declare them once with `pip install rdkit numpy` or via
the `audit` optional extra in `pyproject.toml`):

- `rdkit >= 2025.9.6`
- `numpy >= 2.0`
