"""CardioSafe ensemble loader and forward pass.

Downloads weights from the CardioSafe-benchmark GitHub Releases, caches them
under `~/.cache/cardiosafe/weights/`, loads the 5 seeds of the chosen version,
applies per-checkpoint normalization (descriptor + L1000 z-scoring + reg-head
un-z-scoring), and averages predictions in the natural output space (raw
pIC50 for regression, sigmoid probability for classification).

This module assumes you have already produced a `FeaturizedBatch` via
`inference.featurize.featurize_batch`.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from model.cross_attn import (
    ALL_HEADS,
    CLASSIFICATION_HEADS,
    REGRESSION_HEADS,
    CrossAttnIonChannelPredictor,
)
from model.l1000_encoder import L1000ExpressionEncoder

from .featurize import (
    CHEM_BLOCK_DIM,
    CHEMBERTA_DIM,
    L1000_DIM,
    TOTAL_DIM,
    FeaturizedBatch,
)

CACHE_DIR = Path(os.environ.get("CARDIOSAFE_CACHE", Path.home() / ".cache" / "cardiosafe"))
RELEASE_BASE = "https://github.com/AppliedScientific/CardioSafe-benchmark/releases/download"

VERSION_TAGS = {
    "v1.0": "v1.0-weights",
    "v1.1": "v1.1-weights",
}
L1000_TAG = "l1000-encoder-v1"
L1000_ASSET = "l1000_encoder.pt"
SEEDS = (42, 43, 44, 45, 46)


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def fetch_ensemble_weights(version: str) -> list[Path]:
    """Returns the local paths of the 5 seed weight files for `version`. Downloads
    any missing files from GitHub Releases. Both v1.0 and v1.1 are accepted.
    """
    if version not in VERSION_TAGS:
        raise ValueError(f"Unknown version {version!r}; expected one of {list(VERSION_TAGS)}")
    tag = VERSION_TAGS[version]
    weight_dir = CACHE_DIR / "weights" / version
    paths: list[Path] = []
    for seed in SEEDS:
        asset = f"cardiosafe_{version}_seed_{seed}.pt"
        dest = weight_dir / asset
        if not dest.exists():
            _download(f"{RELEASE_BASE}/{tag}/{asset}", dest)
        paths.append(dest)
    return paths


def fetch_l1000_weights() -> Path:
    dest = CACHE_DIR / "weights" / "l1000" / L1000_ASSET
    if not dest.exists():
        _download(f"{RELEASE_BASE}/{L1000_TAG}/{L1000_ASSET}", dest)
    return dest


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class L1000Wrapper:
    """L1000 encoder + the training-fold per-gene scalers used to recover raw
    expression from the encoder's z-score-space predictions."""
    encoder: L1000ExpressionEncoder
    scaler_means: np.ndarray  # (978,)
    scaler_stds: np.ndarray   # (978,)

    def predict_raw(self, tt1024: np.ndarray) -> np.ndarray:
        """Run the encoder and inverse-scale to raw expression space (the space
        the CardioSafe l1000_norm_means/stds were computed in)."""
        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            z = self.encoder(torch.from_numpy(tt1024).to(device)).cpu().numpy()
        return (z * self.scaler_stds + self.scaler_means).astype(np.float32)


def load_l1000_encoder(device: torch.device | None = None) -> L1000Wrapper:
    """Load the L1000 expression encoder + its scaler onto `device` (defaults to CPU)."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(fetch_l1000_weights(), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = L1000ExpressionEncoder(
        n_genes=cfg["n_genes"],
        mol_dim=cfg["mol_dim"],
        gene_dim=cfg["gene_dim"],
        dropout=cfg["dropout"],
        edge_index=ckpt["edge_index"].to(device),
    ).to(device)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=True)
    encoder.eval()
    return L1000Wrapper(
        encoder=encoder,
        scaler_means=np.asarray(ckpt["scaler_means"], dtype=np.float64),
        scaler_stds=np.asarray(ckpt["scaler_stds"], dtype=np.float64),
    )


@dataclass
class LoadedSeed:
    model: CrossAttnIonChannelPredictor
    descriptor_means: np.ndarray  # (20,)
    descriptor_stds: np.ndarray   # (20,)
    l1000_means: np.ndarray       # (978,)
    l1000_stds: np.ndarray        # (978,)
    reg_scalers: dict[str, tuple[float, float]]  # head_name -> (mean, std)
    config: dict


def load_seed(version: str, seed: int, device: torch.device | None = None) -> LoadedSeed:
    """Load a single seed of the CardioSafe ensemble."""
    if device is None:
        device = torch.device("cpu")
    tag = VERSION_TAGS[version]
    asset = f"cardiosafe_{version}_seed_{seed}.pt"
    path = CACHE_DIR / "weights" / version / asset
    if not path.exists():
        _download(f"{RELEASE_BASE}/{tag}/{asset}", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = CrossAttnIonChannelPredictor(
        chem_dim=cfg["chem_dim"],
        chemberta_dim=cfg["chemberta_dim"],
        bio_dim=cfg["bio_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return LoadedSeed(
        model=model,
        descriptor_means=np.asarray(ckpt["descriptor_scaler_means"], dtype=np.float64),
        descriptor_stds=np.asarray(ckpt["descriptor_scaler_stds"], dtype=np.float64),
        l1000_means=np.asarray(ckpt["l1000_norm_means"], dtype=np.float64),
        l1000_stds=np.asarray(ckpt["l1000_norm_stds"], dtype=np.float64),
        reg_scalers={k: tuple(v) for k, v in ckpt["reg_scalers"].items()},
        config=cfg,
    )


def load_ensemble(version: str, device: torch.device | None = None) -> list[LoadedSeed]:
    fetch_ensemble_weights(version)
    return [load_seed(version, seed, device=device) for seed in SEEDS]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_input(batch: FeaturizedBatch, seed: LoadedSeed, device: torch.device) -> Tensor:
    chem = batch.chem_block.copy()
    chem[:, 6144:6164] = (
        (chem[:, 6144:6164].astype(np.float64) - seed.descriptor_means)
        / np.where(seed.descriptor_stds == 0.0, 1.0, seed.descriptor_stds)
    ).astype(np.float32)

    l1000_z = (
        (batch.l1000_raw.astype(np.float64) - seed.l1000_means)
        / np.where(seed.l1000_stds == 0.0, 1.0, seed.l1000_stds)
    ).astype(np.float32)

    full = np.concatenate([chem, batch.chemberta, l1000_z], axis=1)
    assert full.shape[1] == TOTAL_DIM, f"expected {TOTAL_DIM} dims, got {full.shape[1]}"
    return torch.from_numpy(full).to(device)


def predict(
    batch: FeaturizedBatch,
    ensemble: Iterable[LoadedSeed],
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run the ensemble on a featurized batch. Returns a dict mapping head name to
    per-sample numpy arrays. Regression heads return raw pIC50 (un-z-scored).
    Classification heads return mean sigmoid probability across seeds.
    """
    if device is None:
        device = torch.device("cpu")
    ensemble = list(ensemble)
    if not ensemble:
        raise ValueError("ensemble is empty")

    n = len(batch.smiles)
    reg_accum = {h: np.zeros(n, dtype=np.float64) for h in REGRESSION_HEADS}
    clf_accum = {h: np.zeros(n, dtype=np.float64) for h in CLASSIFICATION_HEADS}

    for seed in ensemble:
        x = _build_input(batch, seed, device=device)
        with torch.no_grad():
            out = seed.model(x)
        for head in REGRESSION_HEADS:
            mean, std = seed.reg_scalers[head]
            pred = out[head].cpu().numpy().astype(np.float64) * std + mean
            reg_accum[head] += pred
        for head in CLASSIFICATION_HEADS:
            prob = torch.sigmoid(out[head]).cpu().numpy().astype(np.float64)
            clf_accum[head] += prob

    n_seeds = len(ensemble)
    return {
        **{h: (reg_accum[h] / n_seeds) for h in REGRESSION_HEADS},
        **{h: (clf_accum[h] / n_seeds) for h in CLASSIFICATION_HEADS},
    }
