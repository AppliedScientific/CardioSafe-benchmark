"""Gradio demo of the CardioSafe ion-channel safety model.

The Space repo intentionally does not vendor the CardioSafe source. At
startup we clone the upstream repos for CardioSafe and MolGpKa, then
prepend them to sys.path. Cache survives across requests; only the first
build pays the clone cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Boot: clone CardioSafe + MolGpKa, wire imports
# ---------------------------------------------------------------------------

CARDIOSAFE_REPO = "https://github.com/AppliedScientific/CardioSafe-benchmark.git"
MOLGPKA_REPO = "https://github.com/Xundrug/MolGpKa.git"

WORK = Path(os.environ.get("CARDIOSAFE_SPACE_WORK", Path.home() / "_cardiosafe")).resolve()
CARDIOSAFE_DIR = WORK / "CardioSafe-benchmark"
MOLGPKA_DIR = WORK / "MolGpKa"


def _ensure_clone(url: str, dest: Path) -> None:
    """Clone the repo, or fast-forward an existing clone to origin/HEAD so the
    Space picks up upstream fixes without a full container rebuild."""
    if dest.exists() and (dest / ".git").exists():
        print(f"  pulling {dest}")
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin"], check=False)
        subprocess.run(["git", "-C", str(dest), "reset", "--hard", "origin/HEAD"], check=False)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


print("Bootstrapping CardioSafe...")
_ensure_clone(CARDIOSAFE_REPO, CARDIOSAFE_DIR)
_ensure_clone(MOLGPKA_REPO, MOLGPKA_DIR)
os.environ["MOLGPKA_SRC"] = str(MOLGPKA_DIR / "src")
sys.path.insert(0, str(CARDIOSAFE_DIR))


def _patch_molgpka_smarts_path() -> None:
    """MolGpKa's utils/ionization_group.py computes `root = abspath(dirname(__file__))`
    at module load. On HF Spaces, __file__ resolves cwd-relative for sys.path
    imports, so `root` and `smarts_file` end up rooted at `/app/utils/` (which
    doesn't exist) and every pKa prediction silently falls back to sentinel
    values. We force-set them to the absolute MolGpKa path after the module
    loads. `get_ionization_aid` reads `smarts_file` from the module global on
    every call, so the patch sticks."""
    import inference.featurize as _fz  # noqa
    _fz._setup_molgpka()  # triggers utils.* imports under MolGpKa/src
    import utils.ionization_group as _uig
    correct_root = str((MOLGPKA_DIR / "src" / "utils").resolve())
    _uig.root = correct_root
    _uig.smarts_file = str(Path(correct_root) / "smarts_pattern.tsv")
    print(f"  patched MolGpKa smarts_file -> {_uig.smarts_file}")


_patch_molgpka_smarts_path()

# ---------------------------------------------------------------------------
# Heavy imports (after sys.path is set)
# ---------------------------------------------------------------------------

import gradio as gr
import pandas as pd
import torch

from inference.ensemble import load_ensemble, load_l1000_encoder, predict  # noqa: E402
from inference.featurize import featurize_batch  # noqa: E402
from model.chemberta_encoder import ChemBERTaEncoder  # noqa: E402
from model.cross_attn import ALL_HEADS  # noqa: E402

DEVICE = torch.device("cpu")
MAX_SMILES = 50

HEAD_LABELS = {
    "herg_pchembl": "hERG pIC50",
    "herg_blocker_10um": "hERG blocker CO (10 µM)",
    "herg_blocker_1um": "hERG blocker CO (1 µM)",
    "nav15_pchembl": "Nav1.5 pIC50",
    "nav15_blocker": "Nav1.5 blocker CO",
    "cav12_pchembl": "Cav1.2 pIC50",
    "cav12_blocker": "Cav1.2 blocker CO",
    "iks_blocker": "IKs blocker CO",
}

# ---------------------------------------------------------------------------
# Eager-load v1.1 ensemble at boot; v1.0 lazy-loaded on first request
# ---------------------------------------------------------------------------

print("Loading ChemBERTa-77M-MTR...")
CHEMBERTA = ChemBERTaEncoder()
print("Loading L1000 encoder...")
L1000 = load_l1000_encoder(device=DEVICE)
print("Loading CardioSafe v1.1 ensemble (5 seeds)...")
ENSEMBLES: dict[str, list] = {"v1.1": load_ensemble("v1.1", device=DEVICE)}
print("Ready.")


def _get_ensemble(version: str):
    if version not in ENSEMBLES:
        print(f"Loading CardioSafe {version} ensemble (5 seeds)...")
        ENSEMBLES[version] = load_ensemble(version, device=DEVICE)
    return ENSEMBLES[version]


def run(smiles_text: str, version: str) -> pd.DataFrame:
    smiles = [s.strip() for s in smiles_text.splitlines() if s.strip()]
    if not smiles:
        raise gr.Error("Enter at least one SMILES.")
    if len(smiles) > MAX_SMILES:
        raise gr.Error(f"Max {MAX_SMILES} SMILES per request. Got {len(smiles)}.")

    ensemble = _get_ensemble(version)
    batch = featurize_batch(smiles, chemberta_encoder=CHEMBERTA, l1000_encoder=L1000)
    preds = predict(batch, ensemble=ensemble, device=DEVICE)

    rows = []
    for i, smi in enumerate(smiles):
        row: dict = {"SMILES": smi}
        for h in ALL_HEADS:
            row[HEAD_LABELS[h]] = round(float(preds[h][i]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


EXAMPLE_INPUT = """CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4
CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4
COc1ccc(CCN2CCC(CC2)Nc3nc4ccccc4n3Cc5ccc(F)cc5)cc1
COc1cc(N)c(Cl)cc1C(=O)N[C@@H]1CCN(CCCOc2ccc(F)cc2)C[C@@H]1OC
CN(CCOc1ccc(NS(=O)(=O)C)cc1)CCc2ccc(NS(=O)(=O)C)cc2
C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@@H](O)c1ccnc2ccc(OC)cc12
COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"""

INTRO_MD = """# CardioSafe — cardiac ion-channel safety predictions

Paste SMILES below (one per line, up to 50) and get predictions for the four
CiPA channels: **hERG, Nav1.5, Cav1.2, IKs** — blocker classification output (CO; sigmoid in [0, 1], not a calibrated probability — the underlying classes are heavily imbalanced), plus pIC50 for hERG / Nav1.5 / Cav1.2 (IKs has no regression head — n = 115 labelled compounds).

This is the paper-snapshot ensemble from
[Jovanović et al. 2026 (bioRxiv)](https://www.biorxiv.org/content/10.64898/2026.05.06.723181v1).
Weights on [HF](https://huggingface.co/appliedscientific/cardiosafe), source on
[GitHub](https://github.com/AppliedScientific/CardioSafe-benchmark).
The continually-updated production model is at
[platform.appliedscientific.ai/cardiosafe](https://platform.appliedscientific.ai/cardiosafe).
"""

FOOTER_MD = """---

**v1.1** is the recommended retrain; it differs from v1.0 by 2 force-routed
analogs in the cardiac-cliff cluster (see
[Note S3](https://github.com/AppliedScientific/CardioSafe-benchmark/blob/main/data/supplementary/note_s3_v1_1_audit_correction.md)).
Test fold and headline metrics are unchanged. **v1.0** is the preprint snapshot —
use it when reproducing paper numbers.

Per-checkpoint normalization, MolGpKa-based pKa descriptors, ChemBERTa-77M-MTR
embeddings, and a learned L1000 expression encoder are all applied automatically.
First request after a cold start may take ~30 s while ChemBERTa is downloaded.
"""

with gr.Blocks(title="CardioSafe", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO_MD)
    with gr.Row():
        with gr.Column(scale=2):
            smiles_in = gr.Textbox(
                label="SMILES (one per line)",
                value=EXAMPLE_INPUT,
                lines=10,
            )
            version_in = gr.Radio(
                ["v1.1", "v1.0"],
                value="v1.1",
                label="Ensemble (v1.1 recommended)",
            )
            btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=3):
            out = gr.DataFrame(label="Predictions", interactive=False, wrap=True)
    gr.Markdown(FOOTER_MD)

    btn.click(run, inputs=[smiles_in, version_in], outputs=out)


if __name__ == "__main__":
    demo.launch()
