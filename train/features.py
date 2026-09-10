r"""Lazy feature assembly for CardioSafe training.

Row-aligned NumPy caches, memory-mapped and gathered per batch rather than materialised:
the full matrix is 334,444 x 6,548 float32, about 8.8 GB, and mmap lets several training
processes share one page cache instead of each holding a copy.

Column layout, matching what the model slices:

    [ morgan 2048 | atompair 2048 | toptorsion 2048 | descriptors 20 | chemberta 384 ]
      \-------------------- chem_dim = 6164 --------------------/

Descriptors are passed in already z-scored, because that scaling has to be fit on the
split's TRAIN rows and this class has no business knowing which those are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FP_DIM = 2048
CHEMBERTA_DIM = 384


class TripleBranchFeatures:
    """Concatenates three fingerprint blocks, descriptors and ChemBERTa on demand."""

    def __init__(
        self,
        morgan_path: Path,
        atompair_path: Path,
        toptorsion_path: Path,
        chemberta_path: Path,
        descriptors_scaled: np.ndarray,
    ) -> None:
        self.morgan = np.load(morgan_path, mmap_mode="r")
        self.atompair = np.load(atompair_path, mmap_mode="r")
        self.toptorsion = np.load(toptorsion_path, mmap_mode="r")
        self.chemberta = np.load(chemberta_path, mmap_mode="r")
        self.descriptors = np.ascontiguousarray(descriptors_scaled, dtype=np.float32)

        blocks = {
            "morgan": self.morgan,
            "atompair": self.atompair,
            "toptorsion": self.toptorsion,
            "chemberta": self.chemberta,
            "descriptors": self.descriptors,
        }
        n_rows = {name: arr.shape[0] for name, arr in blocks.items()}
        if len(set(n_rows.values())) != 1:
            raise ValueError(f"feature arrays disagree on row count: {n_rows}")
        for name in ("morgan", "atompair", "toptorsion"):
            got = blocks[name].shape[1]
            if got != FP_DIM:
                raise ValueError(f"{name} has width {got}, expected {FP_DIM}")
        if self.chemberta.ndim != 2 or self.chemberta.shape[1] < 1:
            raise ValueError(
                "Encoder cache must be a nonempty-width matrix"
            )

        self.n = next(iter(n_rows.values()))
        self.n_desc = self.descriptors.shape[1]
        self.chem_dim = 3 * FP_DIM + self.n_desc
        self.encoder_dim = self.chemberta.shape[1]
        self.width = self.chem_dim + self.encoder_dim

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.width)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: np.ndarray) -> np.ndarray:
        """Gather rows `idx` as one (len(idx), width) float32 array.

        Sort the indices before touching the mmaps: a batch is a random draw over 334k rows
        and ascending access keeps the page reads roughly sequential. The result is restored
        to the caller's order so labels still line up.
        """
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]

        out = np.empty((len(idx), self.width), dtype=np.float32)
        at = 0
        for arr in (self.morgan, self.atompair, self.toptorsion):
            out[order, at : at + FP_DIM] = arr[sorted_idx]
            at += FP_DIM
        out[order, at : at + self.n_desc] = self.descriptors[sorted_idx]
        at += self.n_desc
        out[order, at : at + self.encoder_dim] = self.chemberta[sorted_idx]
        return out


def build_provider(cache_dir: Path, descriptors_scaled: np.ndarray,
                   chemberta_path: Path | None = None) -> TripleBranchFeatures:
    """Open the raw hash-folded 2048-bit fingerprint caches plus ChemBERTa."""
    paths = {
        "morgan": cache_dir / "morgan_fps_2048.npy",
        "atompair": cache_dir / "atompair_fps_2048.npy",
        "toptorsion": cache_dir / "toptorsion_fps_2048.npy",
        "chemberta": chemberta_path if chemberta_path is not None else cache_dir / "chemberta_384.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing feature caches: " + ", ".join(missing) + ". See train/README.md for "
            "what to build and the expected row alignment."
        )
    return TripleBranchFeatures(
        paths["morgan"],
        paths["atompair"],
        paths["toptorsion"],
        paths["chemberta"],
        descriptors_scaled,
    )
