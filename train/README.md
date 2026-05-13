# Training reference

Reference implementations of the losses + training steps described in the
Methods section of Jovanović et al. 2026. **No trained weights are
released.** These files exist so an outside group can verify that the
paper's recipe is internally consistent and re-train a model from the
public data deposit, the published architecture, and their own L1000
prior.

| File | Paper anchor |
| --- | --- |
| `losses.py` | Focal-BCE (Methods: "Stage 1 trained with focal loss, γ = 2.0…") and the margin ranking loss (Methods: "max(0, 1.5 − (logit_blocker − logit_safer))"). |
| `stage1_step.py` | One Stage 1 train step. Per-head NaN masking, focal-BCE on the 5 classification heads, MSE on the 3 regression heads (training-fold z-scored), uniform mean across active heads, Adam(weight_decay=1e-5), grad-clip 1.0. The caller owns data loading, the NoamLR schedule, validation, and early stopping. |
| `stage2.py` | Runnable Stage 2 cliff fine-tune. For each seed 42-46: 9 epochs × 120 mini-batches of 512 ChEMBL rows + the curated cliff rows (cliff weight 8.0); pairwise margin ranking loss with λ = 0.3 on the two hERG heads; AdamW (lr 1e-5, weight_decay 1e-5); grad-clip 1.0. Inputs (feature caches, labels matrix, splits NPZ, cliff NPZ, Stage 1 checkpoints) are all passed as CLI paths. |

## What you need to bring

- **Compounds, splits, labels.** These are already in `data/` of this
  repo. Build your row-aligned label tensor `(N, 8)` and split NPZ
  (with `train`/`val`/`test` row_idx arrays) from them.
- **Featurizer caches.** Six NumPy arrays of length N, row-aligned with
  `data/compounds/compounds.csv`:
  - `morgan_2048.npy` — `(N, 2048)` uint8 / float32
  - `atompair_2048.npy` — `(N, 2048)`
  - `toptors_2048.npy` — `(N, 2048)` (binary, for the chem branch)
  - `descriptors_20.npy` — `(N, 20)` float32 (raw; the script
    z-scores using the checkpoint's training-fold mean/std)
  - `chemberta_384.npy` — `(N, 384)` float32 (run
    `model/chemberta_encoder.py` once over `compounds.csv`)
  - `bio_l1000_978.npy` — `(N, 978)` float32 (run a trained
    `model/l1000_encoder.py` on each compound's *count* TopologicalTorsion
    1024 fingerprint; or substitute your own L1000 prior)
- **Stage 1 checkpoints.** `stage2.py` reads `seed_42/model.pt..seed_46/model.pt`
  + `config.json` from `--stage1-dir`. The checkpoint must carry the
  same scaler statistics keys the deployed code uses
  (`descriptor_scaler_means`, `descriptor_scaler_stds`, `l1000_norm_means`,
  `l1000_norm_stds`, `reg_scalers`).
- **Cliff features NPZ.** A single NPZ with `chem_raw` (N_cliff, 6164),
  `chemberta` (N_cliff, 384), `bio_raw` (N_cliff, 978), `labels`
  (N_cliff, 8), `pair_ids` (N_cliff,), `roles` (N_cliff,). The per-split
  cliff manifests are at `data/supplementary/note_s2_cliff_set_*.csv`.

## Running Stage 2

```bash
python train/stage2.py \
  --seeds 42 43 44 45 46 \
  --epochs 9 \
  --stage1-dir   path/to/stage1_checkpoints \
  --output-dir   path/to/cardiosafe_finetuned \
  --morgan-path  path/to/morgan_2048.npy \
  --atompair-path path/to/atompair_2048.npy \
  --toptors-path path/to/toptors_2048.npy \
  --descriptors-path path/to/descriptors_20.npy \
  --chemberta-path path/to/chemberta_384.npy \
  --bio-path     path/to/bio_l1000_978.npy \
  --labels-path  path/to/labels_v1.npy \
  --splits-path  path/to/splits_tan70.npz \
  --cliff-path   path/to/cliff_features_v6.npz
```

All hyperparameters default to the paper's recipe values (printed in
`--help`). The script writes one checkpoint directory per seed under
`--output-dir`.

## What is *not* shipped

- Trained weights (Stage 1 ensemble, Stage 2 CardioSafe ensemble) —
  inference at [platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe).
- The raw L1000 signatures or their training script — see paper Methods
  and GEO accessions GSE92742 / GSE70138 to rebuild the encoder yourself.
- The Stage 1 trunk's end-to-end runner (data loaders, NoamLR scheduler,
  early-stopping driver). All hyperparameters are in the paper; a clean
  driver around `stage1_step.py` is a 50-line exercise.
- Ablation runners (the Y-randomization, exact-only-curation, L1000
  threshold sweep, bio-zero, ChemBERTa-zero variants). Their headline
  numbers are in supplementary Tables S8, S9 and Note S1 with the data
  needed to recompute them.
