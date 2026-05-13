# CardioSafe model card

Reference architecture from **CardioSafe: multi-task prediction of cardiac
ion channel activity with reverse-leak audited benchmarking** (Jovanović
et al., 2026). The 5-seed paper-snapshot weights for both **v1.0** (the
ensemble evaluated in the preprint) and **v1.1** (the audit-clean retrain
described in Note S3) are distributed via the
[GitHub Releases](https://github.com/AppliedScientific/CardioSafe-benchmark/releases)
of this repository under CC-BY-NC-4.0. See [`../inference/README.md`](../inference/README.md)
for runnable scoring. The continually-updated deployed ensemble is served
at [platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe).

## Inputs

A single flat float tensor of shape `(B, 7526)`. Layout:

| dims | block | source |
| --- | --- | --- |
| 0 – 2047 | Morgan radius-2 2048-bit binary fingerprint | RDKit `GetMorganGenerator(radius=2, fpSize=2048)` |
| 2048 – 4095 | AtomPair 2048-bit binary fingerprint | RDKit `GetAtomPairGenerator(fpSize=2048)` |
| 4096 – 6143 | TopologicalTorsion 2048-bit binary fingerprint | RDKit `GetTopologicalTorsionGenerator(fpSize=2048)` |
| 6144 – 6163 | 20-descriptor block, training-fold z-scored | See `data/supplementary/table_s0_descriptor_spec.*` |
| 6164 – 6547 | ChemBERTa-77M-MTR mean-pooled sentence embedding (384) | `model/chemberta_encoder.py` |
| 6548 – 7525 | L1000 predicted gene-expression z-scores (978) | `model/l1000_encoder.py` |

Total: 6164 + 384 + 978 = 7526.

## Outputs

`forward(x)` returns a `dict[str, Tensor]` with 8 keys, each value a `(B,)` tensor:

| Head | Output | Channel |
| --- | --- | --- |
| `herg_pchembl` | regression — raw pIC50 | hERG |
| `herg_blocker_10um` | logit (apply sigmoid for P) | hERG |
| `herg_blocker_1um` | logit | hERG |
| `nav15_pchembl` | regression — raw pIC50 | Nav1.5 |
| `nav15_blocker` | logit | Nav1.5 |
| `cav12_pchembl` | regression — raw pIC50 | Cav1.2 |
| `cav12_blocker` | logit | Cav1.2 |
| `iks_blocker` | logit | IKs |

IKs has no regression head (n = 115 labelled compounds; treated as exploratory).

## Architecture

```
Branch A   chem features (6164)
           Linear(6164, 512) -> BN -> ReLU -> Dropout(0.2)
           Linear( 512, 256) -> BN -> ReLU -> Dropout(0.2)
           => chem_emb (256)

Branch B   ChemBERTa embedding (384)
           Linear(384, 128) -> LayerNorm -> ReLU -> Dropout(0.2)
           => cb_emb (128)

Branch C   L1000 predicted expression (978)
           Linear(978, 128) -> LayerNorm -> ReLU -> Dropout(0.3)
           => bio_emb (128)

K/V        chem_kv = LayerNorm(Linear(chem_emb, 128))
           cb_kv   = LayerNorm(cb_emb)
           bio_kv  = LayerNorm(bio_emb)
           KV = stack(chem_kv, cb_kv, bio_kv) -> (B, 3, 128)

Queries    Q = nn.Parameter(torch.randn(4, 128) * 0.02)
           4 learnable channel queries: hERG, Nav1.5, Cav1.2, IKs

Attention  MultiheadAttention(embed_dim=128, num_heads=2,
                              dropout=0.1, batch_first=True)
           ctx = attn(Q, KV, KV) -> (B, 4, 128)
           ctx = LayerNorm(ctx)

Heads      For each of the 8 heads:
             head_in = concat(ctx[:, channel, :], chem_emb)  # (B, 128 + 256)
             Linear(384, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1)
```

Weight initialisation: Kaiming-normal (ReLU nonlinearity) on every Linear
weight, biases zero. The four channel queries use the small-scale
Gaussian init shown above.

Parameter count: ~3.96M trainable parameters.

## Featurizers

- **ChemBERTa**: `DeepChem/ChemBERTa-77M-MTR` from the HuggingFace hub,
  frozen, attention-mask-weighted mean pool over `last_hidden_state`.
  See `model/chemberta_encoder.py`.
- **L1000 encoder**: 2 × `GCNConv(384, 384)` over the L1000 gene
  co-expression graph (`|Pearson r| > 0.40` on training-set signatures),
  followed by a shared 3-layer FFN of width 512 and per-gene linear
  heads producing 978 z-score predictions. Input is a 1024-dim count
  TopologicalTorsion fingerprint. See `model/l1000_encoder.py`. The
  graph is computed offline from LINCS Phase I (GSE92742) and Phase II
  (GSE70138); it is not redistributed here.

## What is *not* in this directory

- Trained weights themselves do not live in `model/` — they are
  distributed via the GitHub Releases. See [`../inference/`](../inference/)
  for the loader (auto-downloads + caches under
  `~/.cache/cardiosafe/weights/`).
- A runnable trainer end-to-end. See `../train/` for the loss functions
  (`losses.py`), the Stage 1 train step (`stage1_step.py`), and the
  Stage 2 cliff fine-tune script (`stage2.py`).

## Citing this architecture

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
