---
title: CardioSafe
emoji: ❤️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.6.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
models:
  - appliedscientific/cardiosafe
short_description: hERG/Nav1.5/Cav1.2/IKs safety prediction from SMILES
---

# CardioSafe — interactive demo

Paste SMILES, get predictions for the four CiPA cardiac ion channels:

| Head | Output |
| --- | --- |
| `hERG pIC50`, `Nav1.5 pIC50`, `Cav1.2 pIC50` | raw regression (un-z-scored) |
| `hERG blocker (10 µM / 1 µM)` | probability |
| `Nav1.5 blocker`, `Cav1.2 blocker`, `IKs blocker` | probability |

**v1.1** (audit-clean) is the recommended ensemble. **v1.0** (preprint) is also available for reproduction.

This is the **paper-snapshot** model from
[Jovanović et al. 2026 (bioRxiv)](https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1).
The continually-updated production ensemble — trained on CRO-validated
bioassay data — is served at
[platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe).

- Weights: [`appliedscientific/cardiosafe`](https://huggingface.co/appliedscientific/cardiosafe)
- Source: [`AppliedScientific/CardioSafe-benchmark`](https://github.com/AppliedScientific/CardioSafe-benchmark)
- License: weights CC-BY-NC-4.0; code MIT — see [LICENSE-WEIGHTS](https://github.com/AppliedScientific/CardioSafe-benchmark/blob/main/LICENSE-WEIGHTS).

> Predictions made with this pipeline rely on **MolGpKa**
> (Pan et al. 2021, [doi:10.1021/acs.jcim.1c00075](https://doi.org/10.1021/acs.jcim.1c00075))
> for pKa-derived descriptors, and **ChemBERTa-77M-MTR**
> (Ahmad et al. 2022) for chemical-language embeddings. Please cite both
> if you publish predictions made here.
