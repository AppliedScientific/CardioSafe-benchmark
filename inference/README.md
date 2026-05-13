# CardioSafe inference

Score arbitrary SMILES with the published CardioSafe ensembles.

This directory ships the **paper-snapshot** weights from Jovanović et al.
(2026): both **v1.0** (the version evaluated in the bioRxiv preprint) and
**v1.1** (the audit-clean retrain described in Note S3 — recommended for
new work). The deployed ensemble at
[platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe)
is continually updated with new bioassay validation data routed through
ASI's CRO partner network and may differ from this snapshot. Use the
published weights for paper reproduction and academic benchmarking; use
the platform for active drug-discovery campaigns with CRO-integrated
wet-lab validation.

## Install

```bash
pip install -e ".[inference]"     # from the repo root
```

This pulls in `torch`, `torch-geometric`, `transformers`, `rdkit`,
`pandas`, and `numpy`.

### MolGpKa (required for the 7 pKa-derived descriptors)

7 of the 20 descriptors used by CardioSafe (`pka_acidic`, `pka_basic`,
`logd_7_4`, `frac_cation`, `frac_anion`, `frac_zwitterion`, `frac_neutral`)
are derived from site-level pKa predictions produced by **MolGpka**
(Pan, Wang, Li, Zhang, Ji. *J. Chem. Inf. Model.* **2021**, 61(7),
3159-3165, [doi:10.1021/acs.jcim.1c00075](https://doi.org/10.1021/acs.jcim.1c00075);
MIT-licensed, [github.com/Xundrug/MolGpKa](https://github.com/Xundrug/MolGpKa)).
**Please cite Pan et al. (2021) if you publish predictions made with this
inference pipeline.**

MolGpKa is not on PyPI; clone it and point `MOLGPKA_SRC` at its `src/`
directory:

```bash
git clone https://github.com/Xundrug/MolGpKa.git ~/MolGpKa
export MOLGPKA_SRC=~/MolGpKa/src
```

The 2 GCN weight files (`models/weight_acid.pth`, `models/weight_base.pth`)
ship with the MolGpKa repo — no further setup needed.

## Quickstart

```bash
python -m inference.predict \
    --input inference/example_smiles.csv \
    --output scored.csv \
    --version v1.1
```

The first call downloads the 5 v1.1 ensemble files (~75 MB) and the L1000
encoder (~10 MB) from the GitHub Releases of this repository, cached under
`~/.cache/cardiosafe/weights/`. Subsequent calls are offline.

The output CSV preserves the input columns and appends 8 prediction columns:

| column | meaning |
| --- | --- |
| `cardiosafe_v1.1_herg_pchembl` | hERG pIC50 — raw regression (un-z-scored) |
| `cardiosafe_v1.1_herg_blocker_10um` | hERG @ 10 µM blocker probability |
| `cardiosafe_v1.1_herg_blocker_1um` | hERG @ 1 µM blocker probability |
| `cardiosafe_v1.1_nav15_pchembl` | Nav1.5 pIC50 |
| `cardiosafe_v1.1_nav15_blocker` | Nav1.5 blocker probability |
| `cardiosafe_v1.1_cav12_pchembl` | Cav1.2 pIC50 |
| `cardiosafe_v1.1_cav12_blocker` | Cav1.2 blocker probability |
| `cardiosafe_v1.1_iks_blocker` | IKs blocker probability (exploratory) |

Switch to `--version v1.0` to score with the preprint-version ensemble.

## Programmatic use

```python
import torch
from model.chemberta_encoder import ChemBERTaEncoder
from inference.featurize import featurize_batch
from inference.ensemble import load_ensemble, load_l1000_encoder, predict

device = torch.device("cpu")
chemberta = ChemBERTaEncoder()
l1000 = load_l1000_encoder(device=device)
ensemble = load_ensemble("v1.1", device=device)

smiles = [
    "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",   # terfenadine
    "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",  # fexofenadine
]
batch = featurize_batch(smiles, chemberta_encoder=chemberta, l1000_encoder=l1000)
preds = predict(batch, ensemble=ensemble, device=device)

print("terfenadine hERG pIC50:", preds["herg_pchembl"][0])
print("fexofenadine hERG pIC50:", preds["herg_pchembl"][1])
print("predicted activity cliff:", preds["herg_pchembl"][0] - preds["herg_pchembl"][1])
```

## Cache layout

```
$CARDIOSAFE_CACHE  (defaults to ~/.cache/cardiosafe)
└── weights/
    ├── v1.0/
    │   ├── cardiosafe_v1.0_seed_42.pt    (15 MB)
    │   ├── ... (5 seeds)
    ├── v1.1/
    │   ├── cardiosafe_v1.1_seed_42.pt    (15 MB)
    │   ├── ... (5 seeds)
    └── l1000/
        └── l1000_encoder.pt              (10 MB)
```

## What you get vs the platform

The published ensemble is the **frozen** snapshot used in the paper. The
platform at appliedscientific.ai/cardiosafe additionally provides:

- Continually updated models trained on incoming CRO-validated bioassay data
- Direct CRO booking and routing for top-ranked candidates
- Project-private custom fine-tuning on your assay results
- Hosted batch scoring + REST API + dashboard

For commercial use of the weights themselves, see `LICENSE-WEIGHTS`
(CC-BY-NC-4.0). Contact `business@appliedscientific.ai`.

## Citation

If you use the published weights, please cite the paper and this repository:

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

### Third-party inference dependencies

If you publish predictions made with this pipeline, please also cite the
upstream methods CardioSafe consumes at inference time:

```bibtex
@article{molgpka2021,
  title   = {MolGpka: A Web Server for Small Molecule pKa Prediction
             Using a Graph-Convolutional Neural Network},
  author  = {Pan, Xiaolin and Wang, Hao and Li, Cuiyu and Zhang, John Z. H.
             and Ji, Changge},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {61},
  number  = {7},
  pages   = {3159--3165},
  year    = {2021},
  doi     = {10.1021/acs.jcim.1c00075}
}

@article{chemberta2_2022,
  title   = {ChemBERTa-2: Towards Chemical Foundation Models},
  author  = {Ahmad, Walid and Simon, Elana and Chithrananda, Seyone and
             Grand, Gabriel and Ramsundar, Bharath},
  journal = {arXiv preprint arXiv:2209.01712},
  year    = {2022}
}
```

(The `DeepChem/ChemBERTa-77M-MTR` HuggingFace checkpoint used by
`model/chemberta_encoder.py` is the multi-task-regression variant trained
in Ahmad et al. 2022; the 77M-PubChem pretraining corpus originated in
the earlier Chithrananda et al. 2020 paper, but the MTR-objective
checkpoint specifically is from ChemBERTa-2.)
