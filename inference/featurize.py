"""CardioSafe inference featurizer.

Builds the 7526-dim flat input expected by `model.cross_attn.CrossAttnIonChannelPredictor`
from a list of SMILES.

Layout:
    [Morgan-r2-2048 | AtomPair-2048 | TopologicalTorsion-2048 | desc20 |
     ChemBERTa-384 | L1000-pred-978]
      0 .. 2047       2048 .. 4095    4096 .. 6143             6144 .. 6163
      6164 .. 6547    6548 .. 7525

Per-checkpoint normalization (descriptor z-score, L1000 z-score) is applied
later by `inference.ensemble`, since each saved seed carries its own
training-fold scaler stats.

The 20 descriptors are computed exactly as in
`data/supplementary/table_s0_descriptor_spec.csv`. Descriptors 13-19
(pka_acidic, pka_basic, logd_7_4, frac_cation/anion/zwitterion/neutral)
require MolGpka for site-level pKa prediction:

    Pan, Wang, Li, Zhang, Ji. MolGpka: A Web Server for Small Molecule
    pKa Prediction Using a Graph-Convolutional Neural Network. J. Chem.
    Inf. Model. 2021, 61(7), 3159-3165. doi:10.1021/acs.jcim.1c00075

Please cite Pan et al. (2021) if you publish predictions made with this
pipeline. See `inference/README.md` for setup. Sentinels (acidic=14.0,
basic=0.0) are used when no ionizable site is detected;
Henderson-Hasselbalch derivation of logD_7.4 and microstate fractions
is pure NumPy.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetTopologicalTorsionGenerator,
)

RDLogger.DisableLog("rdApp.*")

CHEM_BLOCK_DIM = 6164
CHEMBERTA_DIM = 384
L1000_DIM = 978
TOTAL_DIM = CHEM_BLOCK_DIM + CHEMBERTA_DIM + L1000_DIM  # 7526

PKA_ACIDIC_SENTINEL = 14.0
PKA_BASIC_SENTINEL = 0.0
PH = 7.4


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

_MORGAN_FP = GetMorganGenerator(radius=2, fpSize=2048)
_ATOMPAIR_FP = GetAtomPairGenerator(fpSize=2048)
_TT_2048_FP = GetTopologicalTorsionGenerator(fpSize=2048)
_TT_1024_COUNT_FP = GetTopologicalTorsionGenerator(fpSize=1024)


def _fp_array(generator, mol: Chem.Mol, n_bits: int) -> np.ndarray:
    return generator.GetFingerprintAsNumPy(mol).astype(np.float32)


def _tt_1024_count(mol: Chem.Mol) -> np.ndarray:
    return _TT_1024_COUNT_FP.GetCountFingerprintAsNumPy(mol).astype(np.float32)


# ---------------------------------------------------------------------------
# 20-descriptor block
# ---------------------------------------------------------------------------

_UNCHARGER = rdMolStandardize.Uncharger()
_pka_models = None  # lazy MolGpKa load


def _setup_molgpka():
    """Import MolGpKa from $MOLGPKA_SRC. Raises RuntimeError with install hints if missing."""
    global _pka_models
    if _pka_models is not None:
        return _pka_models

    molgpka_src = os.environ.get("MOLGPKA_SRC")
    if not molgpka_src:
        raise RuntimeError(
            "MolGpKa is required for the pKa-derived descriptors (cols 13-19). "
            "Set the MOLGPKA_SRC env var to the absolute path of MolGpKa's src/ "
            "directory. See inference/README.md for setup."
        )
    molgpka_root = Path(molgpka_src).resolve()
    if not molgpka_root.exists():
        raise RuntimeError(f"MOLGPKA_SRC does not exist: {molgpka_root}")
    weights_dir = molgpka_root.parent / "models"
    if not (weights_dir / "weight_acid.pth").exists():
        raise RuntimeError(
            f"MolGpKa weights not found at {weights_dir}/weight_acid.pth. "
            "Ensure you cloned the full MolGpKa repo (src/ + models/)."
        )

    if str(molgpka_root) not in sys.path:
        sys.path.insert(0, str(molgpka_root))

    from utils.descriptor import mol2vec  # type: ignore
    from utils.ionization_group import get_ionization_aid  # type: ignore
    from utils.net import GCNNet  # type: ignore

    device = torch.device("cpu")
    model_acid = GCNNet().to(device)
    model_acid.load_state_dict(torch.load(weights_dir / "weight_acid.pth", map_location=device))
    model_acid.eval()
    model_base = GCNNet().to(device)
    model_base.load_state_dict(torch.load(weights_dir / "weight_base.pth", map_location=device))
    model_base.eval()

    _pka_models = (model_acid, model_base, mol2vec, get_ionization_aid, device)
    return _pka_models


def _predict_pka(smiles: str) -> tuple[float, float]:
    """Return (pka_acidic, pka_basic) via MolGpKa. Sentinels if no site detected."""
    model_acid, model_base, mol2vec, get_ionization_aid, device = _setup_molgpka()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return PKA_ACIDIC_SENTINEL, PKA_BASIC_SENTINEL
    try:
        neutral = _UNCHARGER.uncharge(mol)
        if neutral is not None:
            re_mol = Chem.MolFromSmiles(Chem.MolToSmiles(neutral))
            if re_mol is not None:
                mol = re_mol
    except Exception:
        pass
    try:
        mol_h = AllChem.AddHs(mol)
    except Exception:
        return PKA_ACIDIC_SENTINEL, PKA_BASIC_SENTINEL

    def _site_pka(aid: int, model) -> float:
        data = mol2vec(mol_h, aid).to(device)
        with torch.no_grad():
            return float(model(data).cpu().numpy()[0][0])

    try:
        acid_aids = get_ionization_aid(mol_h, acid_or_base="acid")
    except Exception:
        acid_aids = []
    try:
        base_aids = get_ionization_aid(mol_h, acid_or_base="base")
    except Exception:
        base_aids = []

    acid_pkas = []
    for aid in acid_aids:
        try:
            acid_pkas.append(_site_pka(aid, model_acid))
        except Exception:
            continue
    base_pkas = []
    for aid in base_aids:
        try:
            base_pkas.append(_site_pka(aid, model_base))
        except Exception:
            continue
    return (
        min(acid_pkas) if acid_pkas else PKA_ACIDIC_SENTINEL,
        max(base_pkas) if base_pkas else PKA_BASIC_SENTINEL,
    )


def _derive_logd_and_fractions(clogp: float, pka_acidic: float, pka_basic: float) -> tuple[float, float, float, float, float]:
    """Henderson-Hasselbalch closed-form: returns (logd_7_4, frac_cation, frac_anion, frac_zwitterion, frac_neutral)."""
    site_cation = 1.0 / (1.0 + 10.0 ** (PH - pka_basic))
    site_anion = 1.0 / (1.0 + 10.0 ** (pka_acidic - PH))
    frac_cation = site_cation * (1.0 - site_anion)
    frac_anion = (1.0 - site_cation) * site_anion
    frac_zwitterion = site_cation * site_anion
    frac_neutral = (1.0 - site_cation) * (1.0 - site_anion)
    logd = (
        float(clogp)
        - float(np.log10(1.0 + 10.0 ** (pka_basic - PH)))
        - float(np.log10(1.0 + 10.0 ** (PH - pka_acidic)))
    )
    return logd, float(frac_cation), float(frac_anion), float(frac_zwitterion), float(frac_neutral)


def _gasteiger_stats(mol: Chem.Mol) -> tuple[float, float, float, float, float]:
    """Returns (mean, max, min, std, max_positive_n_charge).

    max_positive_n_charge is the largest *positive* Gasteiger partial charge on
    any N atom (0.0 if none are positive, e.g. neutral amines where the partial
    charge on N is typically negative). Filtering for positivity matches the
    training-time featurizer; an earlier version of this function returned the
    max over all N charges, which produced ~0.3 pIC50 drift on neutral-amine
    drugs like terfenadine vs the paper's case-study values.
    """
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array(
        [float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()],
        dtype=np.float64,
    )
    charges = np.where(np.isfinite(charges), charges, 0.0)
    if len(charges) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    positive_n_charges = [
        c for c in (
            float(a.GetDoubleProp("_GasteigerCharge"))
            for a in mol.GetAtoms() if a.GetSymbol() == "N"
        )
        if np.isfinite(c) and c > 0.0
    ]
    max_pos_n = max(positive_n_charges) if positive_n_charges else 0.0
    return (
        float(charges.mean()),
        float(charges.max()),
        float(charges.min()),
        float(charges.std()),
        float(max_pos_n),
    )


def compute_descriptors(smiles: str) -> np.ndarray:
    """Returns the 20-dim RAW descriptor vector for a SMILES. Same column order as
    `data/supplementary/table_s0_descriptor_spec.csv`. Per-checkpoint z-scoring
    is applied later by `inference.ensemble`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20, dtype=np.float64)

    mw = float(Descriptors.MolWt(mol))
    clogp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    rotb = float(Lipinski.NumRotatableBonds(mol))
    arom_rings = float(Lipinski.NumAromaticRings(mol))
    heavy = float(Descriptors.HeavyAtomCount(mol))
    g_mean, g_max, g_min, g_std, max_pos_n = _gasteiger_stats(mol)
    pka_acidic, pka_basic = _predict_pka(smiles)
    logd, fc, fa, fz, fn = _derive_logd_and_fractions(clogp, pka_acidic, pka_basic)

    return np.array(
        [
            mw, clogp, tpsa, hbd, hba, rotb, arom_rings, heavy,
            g_mean, g_max, g_min, g_std, max_pos_n,
            pka_acidic, pka_basic, logd, fc, fa, fz, fn,
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Main featurizer
# ---------------------------------------------------------------------------

@dataclass
class FeaturizedBatch:
    """Raw (pre-normalization) features ready for ensemble forward pass."""
    chem_block: np.ndarray  # (N, 6164) float32 — fingerprints + raw descriptors
    chemberta: np.ndarray   # (N, 384) float32
    l1000_raw: np.ndarray   # (N, 978) float32 — raw L1000 encoder outputs (NOT yet z-scored)
    smiles: list[str]


def featurize_batch(
    smiles_list: list[str],
    chemberta_encoder,
    l1000_encoder,
) -> FeaturizedBatch:
    """Featurize a batch of SMILES into the components needed by the ensemble.

    Args:
        smiles_list: list of canonical SMILES strings.
        chemberta_encoder: instance of `model.chemberta_encoder.ChemBERTaEncoder`.
        l1000_encoder: an `inference.ensemble.L1000Wrapper` (encoder + scaler).
            Its `predict_raw(tt1024)` returns expression in raw (pre-CardioSafe-norm)
            space.
    """
    n = len(smiles_list)
    chem_block = np.zeros((n, CHEM_BLOCK_DIM), dtype=np.float32)
    tt_count_1024 = np.zeros((n, 1024), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        chem_block[i, 0:2048] = _fp_array(_MORGAN_FP, mol, 2048)
        chem_block[i, 2048:4096] = _fp_array(_ATOMPAIR_FP, mol, 2048)
        chem_block[i, 4096:6144] = _fp_array(_TT_2048_FP, mol, 2048)
        chem_block[i, 6144:6164] = compute_descriptors(smi).astype(np.float32)
        tt_count_1024[i] = _tt_1024_count(mol)

    chemberta = chemberta_encoder.encode(smiles_list).astype(np.float32)
    l1000_raw = l1000_encoder.predict_raw(tt_count_1024)

    return FeaturizedBatch(
        chem_block=chem_block,
        chemberta=chemberta,
        l1000_raw=l1000_raw,
        smiles=list(smiles_list),
    )
