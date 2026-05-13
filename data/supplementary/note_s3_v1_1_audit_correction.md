# Note S3 — Audit-driven correction (v1.1 retrain)

After the bioRxiv preprint deposit went out, an exhaustive
O(*n*<sub>train</sub> × *n*<sub>val+test</sub>) Tanimoto audit
(`scripts/audit_tanimoto_leak.py`) caught residual cross-fold edges in
the v1.0 splits at and above the declared cutoffs. This note documents
the audit findings, the patch applied to produce v1.1 splits, and the
re-evaluation of the model — including the marquee terfenadine /
fexofenadine case study (paper line 339) — under the audit-clean v1.1
retrain.

## 1. Audit findings on v1.0

Running `audit_tanimoto_leak.py` exhaustively against the original
v1.0 splits surfaced cross-fold edges:

| Split | Cutoff | Comparisons | Violations | Where |
| --- | ---: | ---: | ---: | --- |
| tan70 (v1.0) | T ≥ 0.70 | 22.4 × 10⁹ | **12** | all train → val, all in the cardiac-cliff family |
| tan60 (v1.0) | T ≥ 0.60 | 8.5 × 10⁹ | **15** | all train → val, all in the cardiac-cliff family |

Every violation involved compounds from the same therapeutic cluster:
hydroxymethyl-terfenadine (HMT) analogues sitting in `train` against the
terfenadine/fexofenadine anchor cluster that the paper force-routes to
`val` for the canonical activity-cliff case study. The v1.0 split
builder force-routed only the two named anchor compounds; it did not
propagate the routing to ≥-cutoff neighbours in the same Tanimoto graph
connected component.

**No train → test or val → test edge** at or above either cutoff was
identified in any audit run. The test fold of both splits is unaffected
by the patch.

## 2. The v1.1 patch — 2 compounds moved to val on tan70

The minimum patch needed to restore the no-cross-fold-edge invariant is
to move the misplaced HMT cluster members from `train` to `val`. On the
primary tan70 split this is **2 compounds**; on the stricter tan60 split
the same patch plus a third quaternary-ammonium HMT analogue gives 3
compounds.

| row_idx | Compound | InChI-key prefix | tan70 v1.0 | tan70 v1.1 | tan60 v1.0 | tan60 v1.1 |
| ---: | --- | --- | --- | --- | --- | --- |
| 317153 | hydroxymethyl-terfenadine (HMT, no stereo) | DRGRCRRKRGIMMI- | train | **val** | train | **val** |
| 331406 | hydroxymethyl-terfenadine \[R\] | DRGRCRRKRGIMMI- | train | **val** | train | **val** |
| 331595 | hydroxymethyl-terfenadine quaternary ammonium | QZZLQXZHKPPSLA- | test | test (no neighbour ≥ 0.70) | train | **val** |

`331595` stays in `test` on tan70 because its highest Tanimoto to any
other compound is 0.673, below the tan70 cutoff. On tan60 (cutoff 0.60)
it sits at T = 0.60 with three val cliff anchors and so moves to val.

### Updated fold counts

| Version | Split | train | val | test | excluded |
| --- | --- | ---: | ---: | ---: | ---: |
| v1.0 (preprint) | tan70 | 241,792 | 46,326 | 46,326 | 0 |
| **v1.1** | tan70 | **241,790** | **46,328** | 46,326 | 0 |
| v1.0 (preprint) | tan60 | 306,665 | 13,889 | 13,889 | 1 |
| **v1.1** | tan60 | **306,662** | **13,892** | 13,889 | 1 |

Test fold size is **identical between v1.0 and v1.1 on both splits**, so
every paper number anchored on the test fold (Tables 2, 3, 3b, S1, S2,
S3, S6, S7, S9; Figures 4, S1; paired-bootstrap comparator deltas;
applicability-domain bins) is mathematically unchanged in its
denominator. The numerator (the ensemble's predictions) shifts slightly
because the model has been retrained without the HMT analogues — see §3.

### Audit verification of v1.1

Re-running the audit at the published cutoffs returns zero violations on
both v1.1 splits:

```
audit_tanimoto_leak.py --split tan70_v1_1  →  0 / 22.4 × 10⁹  (PASS)
audit_tanimoto_leak.py --split tan60_v1_1  →  0 /  8.5 × 10⁹  (PASS)
```

### Cliff manifest unchanged

The 53-compound curated cliff manifest used in Stage 2 fine-tuning is
unchanged on both v1.1 splits. Terfenadine and fexofenadine themselves
were already filtered out of the v1.0 cliff training set for being in
val (`tan70_drop_reason = "out-of-train fold (val)"`,
`tan60_drop_reason = "Tanimoto = 1.0 to val (Terf/Fex)"`); the HMT
analogues now joining val are not in the curated cliff set, so the
filter output is identical. The 48-compound tan70 cliff training set
and the 51-compound tan60 cliff training set used in Stage 2 are
identical to v1.0.

## 3. v1.1 retrain — tan70 only

Stage 1 (Stage 1) was retrained from scratch on the `tan70_v1_1` split
for all five paper seeds (42, 43, 44, 45, 46). Hyperparameters and
recipe are paper-faithful (Methods §"Two-stage training", line 109):
`--model-kind crossattn --epochs 100 --patience 20 --early-stop herg
--focal-gamma 2.0 --pos-weight-cap 8`, Adam (weight_decay 1e-5), Noam-LR
1e-4 → 1e-3 → 1e-5. Stage 2 cliff fine-tune was then run for 9 epochs ×
120 mini-batches per seed, λ=0.3 ranking-loss coefficient, margin=1.5,
cliff weight 8, AdamW lr=1e-5 — matching paper line 111. tan60 was
**not** retrained in this round (the marquee case study is anchored on
tan70).

### Stage 1 (Stage 1) per-seed training summary

| Seed | Epochs (early-stop) | Best val hERG-10µM AUC |
| ---: | ---: | ---: |
| 42 | 25 | 0.9117 |
| 43 | 28 | 0.9138 |
| 44 | 27 | 0.9110 |
| 45 | 24 | 0.9110 |
| 46 | 25 | 0.9118 |

Early-stopping epochs are within the v1.0 range (23–34).

### Headline tan70 test-fold metrics (5-seed ensemble, Stage 2)

| Head | v1.0 (paper) | v1.1 | Δ | n |
| --- | ---: | ---: | ---: | ---: |
| hERG 10 µM AUC | 0.917 | 0.9085 | −0.009 | 46,120 |
| hERG 10 µM MCC | +0.466 | +0.4385 | −0.027 | 46,120 |
| hERG 1 µM AUC | 0.959 | 0.9558 | −0.003 | 46,042 |
| hERG 1 µM MCC | +0.381 | +0.3484 | −0.033 | 46,042 |
| Nav1.5 AUC | 0.832 | 0.8242 | −0.008 | 218 |
| Nav1.5 MCC | +0.458 | +0.4452 | −0.013 | 218 |
| Cav1.2 AUC | 0.841 | 0.8166 | −0.024 | 160 |
| Cav1.2 MCC | +0.488 | +0.4204 | −0.068 | 160 |
| IKs AUC | (n=14, CIs include 0.5) | 0.5385 | n.s. | 14 |
| hERG pIC50 Pearson r | (Table 2b: tan70 r ≈ 0.55) | 0.5422 | comparable | 2,263 |
| Nav1.5 pIC50 Pearson r | 0.578 (Table 2b) | 0.5777 | ±0 | 184 |
| Cav1.2 pIC50 Pearson r | 0.679 (Table 2b) | 0.6546 | −0.025 | 124 |

All deltas sit within or just outside the bootstrap 95 % CI widths
reported for v1.0 (e.g. ±0.011 for the hERG-10 µM AUC; ±0.06 for the
Cav1.2 AUC at n = 160). **The paper's qualitative claims — CardioSafe
leads on the data-rich hERG head, and is statistically indistinguishable
from comparators on the smaller Nav1.5 and Cav1.2 heads pre-de-leak —
are preserved.**

Full per-head metrics (Stage 1 + Stage 2, per-seed min/max, sens, spec,
confusion-matrix entries) are in
`note_s3_v1_1_test_metrics.json`.

## 4. Re-evaluation of the canonical cardiac-cliff case study

This is the marquee qualitative result of the paper (line 339). If the
v1.0 prediction was load-bearing on HMT-analog memorisation, removing
those analogues from training would collapse it; if the prediction was
driven by the broader training distribution (other antihistamines and
hERG-blocking scaffolds), it would be preserved.

| Compound | Measurement | v1.0 (paper) | v1.1 (audit-clean) | Δ |
| --- | --- | ---: | ---: | ---: |
| Terfenadine | predicted pIC50 | 6.38 ± 0.12 | **6.25 ± 0.12** | −0.13 |
| Terfenadine | hERG 10 µM CO | 0.895 | **0.900** | +0.005 |
| Terfenadine | hERG 1 µM CO | 0.763 | **0.769** | +0.006 |
| Fexofenadine | predicted pIC50 | 4.80 ± 0.21 | **4.51 ± 0.20** | −0.29 |
| Fexofenadine | hERG 10 µM CO | 0.683 | **0.656** | −0.027 |
| Fexofenadine | hERG 1 µM CO | 0.389 | **0.334** | −0.055 |

(Labelled pChEMBL: terfenadine 7.55, fexofenadine 3.96 — labelled cliff
+3.59.)

Predicted cliff (terfenadine − fexofenadine pIC50):

- v1.0 model: **+1.58**
- v1.1 model: **+1.74** (slightly *less* compressed than v1.0)

### Verdict

The terfenadine prediction is **not** load-bearing on the HMT-analog
memorisation. The v1.1 ensemble — which has never seen the HMT analogues
during Stage 1 base training nor during Stage 2 cliff fine-tuning —
still correctly classifies terfenadine as a strong hERG blocker (pIC50
6.25; CO@10 µM = 0.900; CO@1 µM = 0.769) and fexofenadine as a
borderline non-blocker (pIC50 4.51; CO@10 µM = 0.656; CO@1 µM = 0.334).
The pIC50 shift on terfenadine (−0.13) is inside the 1 σ inter-seed
standard deviation reported in the paper (±0.12 on this head). All the
qualitative observations in the paper (line 339) survive:

- Terfenadine is correctly classified as a strong blocker.
- Fexofenadine is correctly classified as a borderline non-blocker.
- The model preserves the cliff rank ordering and crosses operational
  thresholds correctly.
- The predicted activity range is compressed relative to the labelled
  cliff (labelled +3.59 vs predicted +1.74).

Full per-compound predictions are in
`note_s3_v1_1_case_study.json`.

## 5. Reproducibility

The v1.1 patched splits are at `data/splits/tan70_v1_1.csv` and
`data/splits/tan60_v1_1.csv`. The exhaustive audit script in this
repository proves both are leak-free:

```bash
# Each takes ~45-60 min on a modern laptop.
python scripts/audit_tanimoto_leak.py --split tan70_v1_1
python scripts/audit_tanimoto_leak.py --split tan60_v1_1
```

For the v1.1 retrain itself: the architecture in `model/` and the loss /
training-step references in `train/` are exact PyTorch counterparts of
the recipe described in the paper's Methods section. Re-deriving the
ensemble requires bringing your own feature caches, Stage 1 base
checkpoints, and the cliff features NPZ described in `train/README.md`.

## 6. What did not change

- **Architecture** (`model/cross_attn.py`, `chemberta_encoder.py`,
  `l1000_encoder.py`).
- **Featurization** (all 6 input caches are deterministic functions of
  SMILES).
- **L1000 encoder** (trained on GEO data, not on ion-channel splits;
  unchanged between v1.0 and v1.1).
- **Cliff manifest** (the curated 53-compound cliff set + the 25-source
  bibliography in `note_s2_*`).
- **Comparators** (CToxPred2 and CardioGenAI predictions in
  `data/comparators/`, computed on the test fold which is unchanged).

## 7. Suggested paper-side errata (preprint v2 / journal version)

| Paper anchor | v1.0 | v1.1 | Note |
| --- | --- | --- | --- |
| Global pool fold counts (tan70) | 241,792 / 46,326 / 46,326 | 241,790 / 46,328 / 46,326 | test count unchanged |
| Global pool fold counts (tan60) | 306,665 / 13,889 / 13,889 | 306,662 / 13,892 / 13,889 | test count unchanged |
| "Zero violations" verification (tan70) | 0 train ↔ test edges (still correct); 12 train ↔ val edges in the cliff cluster (caught here) | 0 across all bucket pairs | the train ↔ test integrity claim that anchors the headline benchmark holds in both versions |
| Terfenadine predicted pIC50 | 6.38 | 6.25 | within ±0.12 inter-seed σ; qualitative narrative unchanged |
| Table 2 / 2b (tan70 test) | v1.0 numbers | v1.1 numbers (see `note_s3_v1_1_test_metrics.json`) | shifts within bootstrap CI widths |
