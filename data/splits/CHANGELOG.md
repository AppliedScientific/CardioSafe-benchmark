# Splits CHANGELOG

## v1.1 (2026-05-13)

`tan70_v1_1.csv`, `tan60_v1_1.csv` — patched splits, audit-clean.

### Motivation

An exhaustive O(n_train × n_other) Tanimoto audit
(`scripts/audit_tanimoto_leak.py`, full O(n²) sweep on both splits)
identified residual cross-fold edges in the v1.0 splits at or above the
declared cutoffs:

- **tan70 v1.0**: 12 train → val edges at T ≥ 0.70.
- **tan60 v1.0**: 15 train → val edges at T ≥ 0.60.

Every violation involves the same therapeutic family: hydroxymethyl-terfenadine
(HMT) analogues in `train` vs. the terfenadine/fexofenadine stereo-cluster
that the paper force-routes to `val` as the canonical cardiac-cliff case
study. The v1.0 split builder force-routed only the two anchor compounds
(terfenadine + fexofenadine) and did not propagate the routing to their
≥-cutoff neighbours.

**Test fold is unaffected in both splits.** Every leak is train → val; no
train → test or val → test edge was found at or above either cutoff.

### Patch

| Move | row_idx | Compound | tan70 v1.0 | tan60 v1.0 | tan70 v1.1 | tan60 v1.1 |
| --- | ---: | --- | --- | --- | --- | --- |
| train → val | 317153 | hydroxymethyl-terfenadine (no stereo) | train | train | **val** | **val** |
| train → val | 331406 | hydroxymethyl-terfenadine \[R\] | train | train | **val** | **val** |
| train → val | 331595 | hydroxymethyl-terfenadine quaternary ammonium | test | train | test | **val** |

`331595` stays in `test` on tan70 because its highest Tanimoto to any
other compound is 0.673 < 0.70 — below the tan70 cutoff, so no constraint
violation. On tan60 (cutoff 0.60) it sits at T = 0.60 with three val
cliff anchors, so it moves to val.

### Fold counts

| Split | Fold | v1.0 | v1.1 | Δ |
| --- | --- | ---: | ---: | ---: |
| tan70 | train | 241,792 | 241,790 | −2 |
| tan70 | val   | 46,326  | 46,328  | +2 |
| tan70 | test  | 46,326  | 46,326  | 0 |
| tan60 | train | 306,665 | 306,662 | −3 |
| tan60 | val   | 13,889  | 13,892  | +3 |
| tan60 | test  | 13,889  | 13,889  | 0 |
| tan60 | excluded | 1 | 1 | 0 |

### Audit verification

After the patch, the full O(n²) audit passes on both splits at the
declared cutoffs. A targeted cluster check confirms all 9 members of the
canonical terfenadine/fexofenadine/HMT cluster now live in `val` (except
`331595` on tan70, which has zero T ≥ 0.70 neighbours).

```bash
python scripts/audit_tanimoto_leak.py --split tan70_v1_1
python scripts/audit_tanimoto_leak.py --split tan60_v1_1
```

### Cliff manifest impact

None. Both terfenadine and fexofenadine were already dropped from the
v1.0 cliff training set (`note_s2_per_split_manifest.csv`,
`tan70_drop_reason = "out-of-train fold (val)"`,
`tan60_drop_reason = "Tanimoto = 1.0 to val (Terf/Fex)"`). The three HMT
analogues joining val in v1.1 do not filter any new compounds out: the
48-compound tan70 cliff set and the 51-compound tan60 cliff set are
unchanged.

### Paper-side impact

| Paper anchor | Number | v1.0 | v1.1 | Headline impact |
| --- | --- | ---: | ---: | --- |
| Line 73 | tan70 train | 241,792 | 241,790 | one-sentence errata |
| Line 73 | tan70 val   | 46,326  | 46,328  | one-sentence errata |
| Line 73 | tan70 test  | 46,326  | 46,326  | unchanged |
| Line 125 | "zero violations" (tan70) | (false in v1.0) | (true in v1.1) | sentence becomes accurate after retrain on v1.1 splits |
| Line 127 | tan60 test  | 13,889  | 13,889  | unchanged |
| Line 339 | terfenadine pIC50 prediction (6.38) | based on v1.0 train | requires retrain on v1.1 train | the marquee case study is materially affected — Stage 1 + Stage 2 retraining is required |

### v1.0 retained

The v1.0 split CSVs are retained at `data/splits/tan70.csv` and
`data/splits/tan60.csv` so the paper's first-pass numbers remain
reproducible. Future evaluations should use the `_v1_1` variants.
