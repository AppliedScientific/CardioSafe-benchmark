# CardioSafe revision: code companion for Lukas

This branch accompanies Mihailo's two original ZIP archives and `delivery_manifest.json`,
plus the completed baseline addendum ZIP and its separate JSON manifest.
It contains the model, training, feature extraction, analysis and plotting code.
The large data, checkpoints and paper materials come from the archives.

**Locked architecture:** three 2048-bit chemical fingerprints + 20 descriptors +
the final ChemBERTa-MTR `<s>` representation (384) + its 199 pretrained predictions.
The three branch projections (256 + 64 + 128) are concatenated. Eight output heads
cover hERG, NaV1.5, CaV1.2 and exploratory IKs. There is no L1000, layer2 readout or
attention in this final model. Research attention modules remain dependencies of
shared experiment infrastructure; their presence does not change this contract.

**Scientific status:** five final seeds on each of tan70/tan60 are complete.
All 160 additional reviewer-baseline fits are complete: 320 checkpoint/fold replays
and 64 ensemble checks passed. Both compute instances are stopped with reusable
features retained. The author authorized proceeding despite whole-row randomized-label
ensembles retaining real hERG ranking (AUROC 0.5667 / 0.6236); its cause remains unresolved.
IKs has only one TEST positive per split. The paper material is a handback draft,
not a submission-ready revision or proof of branch superiority.

## Get this branch

In an existing clone with a clean working tree:

```bash
git fetch origin
git switch --track origin/codex/lukas-final-architecture
```

Or clone it directly:

```bash
git clone --branch codex/lukas-final-architecture --single-branch \
  https://github.com/AppliedScientific/CardioSafe-benchmark.git
cd CardioSafe-benchmark
```

## Import the companion artifacts

Place the two original ZIPs, `delivery_manifest.json`, the baseline addendum ZIP,
and `baseline_addendum_manifest_2026-09-11_004644.json` together outside the repository.
The addendum filename is `cardiosafe_baseline_addendum_2026-09-11_004644.zip`.
Its SHA-256 is in the separate manifest's `archive.sha256` field; the original
manifest uses an `archives` list. The original ZIPs are unchanged.
Check each ZIP's SHA-256 against its entry in the JSON manifest (`shasum -a 256`
on macOS, `sha256sum` on Linux). The archives also contain per-member hash manifests.

From the repository root, replace `/path/to/handback` below with that directory.
Extract data and reports from the archives while keeping this branch's code:

```bash
unzip -o /path/to/handback/cardiosafe_paper_completed_material_2026-09-10.zip \
  'reports/paper_revision/*' 'data/*' 'runs_revision*/*'
unzip -o /path/to/handback/cardiosafe_final_v3_models_2026-09-10.zip \
  'reports/paper_revision/*' 'data/*' 'runs_revision_v3/*'
unzip -o /path/to/handback/cardiosafe_baseline_addendum_2026-09-11_004644.zip \
  'reports/paper_revision/*' 'runs_revision*/*'
```

Extract the addendum last: it supersedes the original archive's paused-baseline
status. Start with `reports/paper_revision/BASELINE_ADDENDUM.md`,
`notes/baseline_final_results.md`, `notes/baseline_per_head_handback.md`,
`notes/reviewer_baseline_response.md`, and `requirements_matrix.md`.
The original `DELIVERY_STATUS.md` and older paused/running notes are historical. The Methods,
reviewer responses, captions, tables and finished figures are there. Historical
material in the original root README and `data/supplementary` must not be mistaken
for the revised model/results. Use only `labels_v3.csv` for this revision. The
historical v1 labels on main have the previously identified hERG Central sign
problem and must not be used for the final model. No label CSV updates are
published on this code branch; revised labels come from the handback archive.

## Environment

Run modules from the repository root. Python 3.13 and the core versions in
`requirements-revision.txt` were used for the local checks; a fresh environment is
recommended. This is a scientific working environment, not a portable lockfile
covering every platform or optional external comparator tool.

```bash
python3.13 -m venv .venv-revision
source .venv-revision/bin/activate
python -m pip install -r requirements-revision.txt
python -m pytest -q tests/test_revision_*.py tests/test_mtr_predictions.py \
  tests/test_fusion_objectives.py tests/test_readout_contract.py
```

The curation-difference integration test additionally needs the historical
`labels_v2.csv`, which is not in the two ZIPs; it explicitly skips when those
optional audit inputs are absent. It passed locally with both versioned datasets.
The other revision tests do not require that historical CSV.

UMAP plots use `umap-learn==0.5.12`; tree baselines use
`xgboost==3.2.0`. Install these optional packages when using those modules.
Mordred and external comparator inference have additional pinned
asset/runtime contracts in the supplied report. Some source-reconstruction audits
expect the sibling `asi` repository and its original source caches. The completed
curation evidence and source snapshots are already supplied in the paper archive.
RF training used scikit-learn 1.9.0; sampled saved-model replays passed with the
local 1.8.0 environment. Prefer the original training version for pickle loading.
The analysis ZIP includes all baseline prediction arrays and provenance, but not
baseline comparison weights or bulk features. Checkpoint replay therefore needs
the separately retained weights/caches; metric regeneration does not.
Cloud launch/watch/shutdown drivers are not part of this handoff workflow.

## Use saved results without rebuilding features

The archives contain final, control and baseline score arrays, so metrics and ROC/PR plots
can be regenerated without the multi-gigabyte feature cache:

```bash
python -m scripts.revision_metrics \
  --arm final=runs_revision_v3/final/tan70_v1_1 \
  --reference final --split tan70_v1_1 --fold test \
  --out reports/paper_revision/tables/lukas_final_tan70_test.csv
python -m scripts.draw_revision_performance
```

To regenerate the complete baseline tables and six figures after importing all
three ZIPs:

```bash
python -m scripts.revision_baseline_metrics
python -m scripts.draw_revision_baseline_results
```

The baseline metric command also writes direct paired fine-tuned versus matched
frozen-control comparisons, separately from comparisons against the locked final.
`write_revision_baseline_handback` additionally requires the included complete
checkpoint-audit JSON. It uses the full table matrix, not a partial run.

Replace tan70 with tan60 for the second final-model metric table. This uses 1,000 molecule
bootstrap draws; intervals condition on the already fitted ensemble.

## Model inference and future retraining

`model/revision.py:RevisionConcat` implements the eight-output model.
`inference/revision_predictor.py:RevisionEnsemble` loads the five checkpoints for
one split plus `data/features/revision_<split>/manifest.json` and validates their
contracts. `predict_features(chemical, cls, mtr_predictions)` expects aligned
N×6164 raw chemical inputs, N×384 final-layer initial-token vectors, and N×199
pretrained predictions after the authors' denormalization, before CardioSafe's
TRAIN scaling. It returns mean sigmoid scores and mean unscaled pChEMBL.

**The ZIPs do not contain the bulk feature arrays or a validated new raw-SMILES
serving pipeline.** Feature replay/training additionally requires Mihailo's retained
`data/features` cache: the three `*_fps_2048.npy` files, `descriptors_20.npy`,
`chemberta_mtr_readouts`, `chemberta_mtr_cls_predictions`, and both
`revision_tan{70,60}_v1_1` directories, including their manifests. Do not substitute
the historical 384-dimensional mean embedding or missing-descriptor sentinels.
The pretrained model/readout and cache hashes are recorded in the supplied
architecture lock and release manifest. See `notes/release_integration.md`.

With those caches available, the verified feature-adapter replay is:

```bash
python -m scripts.verify_revision_release_inference
```

For reference, the explicit final training recipe for one seed is below. Additional
baseline experiments are complete. This command is for deliberate reproduction,
not an instruction to start another training campaign.

```bash
TORCH_THREADS=1 python -m train.train_revision_baselines \
  --out runs_revision_reproduction/final/tan70_v1_1/seed_42 \
  --seed 42 --split tan70_v1_1 --labels data/labels/labels_v3.csv \
  --chemberta-cache data/features/revision_tan70_v1_1/combined.npy \
  --attention none --loss pooled --selection wrmse \
  --descriptor-transform signed_log
```

The code is the reviewed current working source. Some training helpers changed
since the final fits; original source hashes remain in each checkpoint's provenance.
Do not claim this branch is an exact historical training snapshot. The independent
raw-feature inference replay verifies the final model's current loading behavior.
