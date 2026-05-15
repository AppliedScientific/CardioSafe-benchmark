# Changelog

All notable changes to CardioSafe-benchmark. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [v1.1.0] — 2026-05-15

First tagged release. Captures the v1.1 audit-clean snapshot as of paper
deposit, plus three companion HuggingFace artifacts so the model is
discoverable, mirrored, and runnable end-to-end without cloning the
GitHub repo.

Companion HuggingFace releases:
- 🧠 Model — [`appliedscientific/cardiosafe`](https://huggingface.co/appliedscientific/cardiosafe):
  5-seed v1.0 and v1.1 ensembles + L1000 expression encoder. CC-BY-NC-4.0.
- 📊 Dataset — [`appliedscientific/cardiosafe-benchmark`](https://huggingface.co/datasets/appliedscientific/cardiosafe-benchmark):
  curated compounds, sparse 8-task labels, four Tanimoto-controlled split
  configs (`tan70_v1_0`, `tan60_v1_0`, `tan70_v1_1`, `tan60_v1_1`), and
  the verbatim supplementary materials. CC-BY-4.0.
- ❤️‍🩹 Space — [`appliedscientific/cardiosafe`](https://huggingface.co/spaces/appliedscientific/cardiosafe):
  Gradio demo, free CPU tier. Paste SMILES, get hERG/Nav1.5/Cav1.2/IKs
  predictions in the browser.

### Added

- **Three HF artifacts** with reproducible publish scripts:
  `scripts/publish_to_hf.py` (model weights),
  `scripts/publish_space_to_hf.py` (Gradio demo),
  `scripts/publish_dataset_to_hf.py` (dataset). All read `HF_TOKEN` from
  `.env` and are idempotent — safe to re-run when artifacts evolve.
- **Gradio app** (`space/app.py`) clones `CardioSafe-benchmark` +
  `MolGpKa` at boot and prepends them to `sys.path`, so the Space stays
  thin and follows upstream fixes via `git fetch --depth 1 origin && git
  reset --hard origin/HEAD` on every restart. Per-checkpoint
  normalization, MolGpKa-based pKa descriptors, ChemBERTa-77M-MTR
  embeddings, and the learned L1000 encoder are all applied
  automatically.
- **Dataset packaging**: four pre-joined per-fold Parquet shards
  (compounds + labels + fold assignment) surfaced as HF named configs,
  so users can do
  `load_dataset("appliedscientific/cardiosafe-benchmark", "tan70_v1_1", split="test")`
  and get a 46,326-row test fold with all 8 label columns. Raw canonical
  Parquets shipped under `raw/` for power users doing custom merges.

### Fixed

- **`inference/featurize.py` — `_gasteiger_stats` `maxPosN` bug.** The
  function was returning `max` over **all** N-atom Gasteiger charges
  instead of `max` over **positive** N charges. The training-time
  featurizer filtered for `c > 0` and stored `0.0` when no N was
  positively charged, so descriptor dim 12 silently diverged from the
  training pipeline for every neutral-amine molecule. This produced ~0.3
  pIC50 drift on the published case-study compounds — terfenadine came
  out at 5.92 vs the paper's 6.25. Verified bit-equal against the
  authoritative training featurizer after the fix; the live HF Space
  now reproduces the Note S3 case-study values: **terfenadine pIC50
  6.248, fexofenadine pIC50 4.512, cliff +1.736**.
- **HF Space — MolGpKa `smarts_file` path resolution.** MolGpKa's
  `utils/ionization_group.py` computes
  `root = os.path.abspath(os.path.dirname(__file__))` at module load. On
  HF Spaces the loader sets `__file__` to a cwd-relative path
  (`utils/ionization_group.py`), so `root` resolved to `/app/utils`,
  `smarts_file` pointed at a non-existent `/app/utils/smarts_pattern.tsv`,
  and **every pKa prediction silently fell back to sentinel values
  (acidic=14.0, basic=0.0)** with no error. `_patch_molgpka_smarts_path()`
  in `space/app.py` resets the module globals to the resolved absolute
  MolGpKa path after `_setup_molgpka()` loads them; `get_ionization_aid`
  reads `smarts_file` from the module global on each call, so the patch
  sticks.
- **HF Space — Python and `torch_scatter` pins.** Spaces' default
  Python 3.13 dropped `audioop` from stdlib, which breaks gradio's
  `pydub` transitive import. Pinned `python_version: "3.11"` in
  `space/README.md`. Also pinned `torch==2.6.0` and added
  `torch-scatter` from the PyG wheel index (`--find-links
  https://data.pyg.org/whl/torch-2.6.0+cpu.html`) because Spaces' base
  image installed torch 2.12 by default and PyG has no `torch_scatter`
  wheel for 2.12 (source build fails, killing MolGpKa's GCN at first
  predict call).
- **HF Space — `gradio` ↔ `huggingface_hub` incompatibility.** gradio
  5.x still does `from huggingface_hub import HfFolder`, which was
  removed in `huggingface_hub` 1.0. Pinned `huggingface_hub>=0.24,<1.0`
  in `space/requirements.txt`.

### Changed

- **Classifier output labelled `CO`, not `P`.** The 5 binary heads
  return raw sigmoid outputs of a heavily class-imbalanced classifier —
  calling them "probability" implied calibration that the model does
  not have. The Space's table headers, the Space's intro copy, and the
  Space's README now use `CO` (classification output) instead of `P`
  and explicitly note "sigmoid in [0, 1], not a calibrated probability".
- **Toned down "audit-clean" framing.** The audit relocated 2 cardiac-cliff
  analogs out of ~250k training rows between v1.0 and v1.1, with the
  test fold identical across both versions. The previous copy implied
  the change was more material than it actually is. v1.1 is now
  consistently framed as "recommended retrain; differs from v1.0 by 2
  force-routed analogs in the cardiac-cliff cluster — see Note S3" and
  the radio label is `Ensemble (v1.1 recommended)`.
- **"pIC50 (where applicable)" replaced with an explicit note** that IKs
  has no regression head (n=115 labelled compounds; treated as
  exploratory).

### Known limitations / deferred to v1.1.1+

- **`space/app.py`'s MolGpKa monkey-patch is structurally fragile.** If
  MolGpKa upstream ever renames `root` or `smarts_file` module globals,
  the patch silently no-ops and predictions revert to sentinel pKa
  values without a visible error. A safer fix is upstream in
  `inference/featurize.py` (compute `smarts_file` lazily from
  `MOLGPKA_SRC` at first call, not from `__file__` at module load), then
  retire the Space patch.
- **No CI on the HF Space.** A breaking change to upstream MolGpKa or to
  HF's Python base image won't be caught until a user hits the live URL.
  A smoke test (POST to `/gradio_api/call/run` with terfenadine + assert
  `pIC50 == 6.248 ± 0.01`) would close this gap.
- **CardioSafe weights still live on GitHub Releases.** The
  `inference/ensemble.py` loader fetches from the
  `AppliedScientific/CardioSafe-benchmark` Releases page, not from the
  HF model repo. HF is a discoverability mirror today, not the primary
  source. Migrating the loader to `huggingface_hub.hf_hub_download`
  would unify the artifacts; deferred so we don't break the current
  inference path in this release.
