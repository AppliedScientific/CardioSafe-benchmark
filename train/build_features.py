"""Build the training feature caches from `data/compounds/compounds.csv`.

Writes five row-aligned NumPy arrays that `train/features.py` memory-maps:

    morgan_fps_2048.npy      (N, 2048) float32
    atompair_fps_2048.npy    (N, 2048) float32
    toptorsion_fps_2048.npy  (N, 2048) float32
    descriptors_20.npy       (N,   20) float32
    chemberta_384.npy        (N,  384) float32

Row `i` of every array is `row_idx == i` in `compounds.csv`, so all of them index-join
against the labels and splits without a lookup. That alignment is the whole contract: an
off-by-one permutation here would not crash anything, it would just quietly train the model
on other compounds' labels.

The fingerprints and descriptors come from `inference/featurize.py` rather than being
reimplemented, so training and inference cannot drift apart in how a molecule is encoded.

MolGpKa is required -- 7 of the 20 descriptors are ionization-derived. If it is missing or
broken, `_predict_pka` falls back to sentinels and this script will warn loudly; a run that
warns has produced degraded descriptors and should be discarded. See `inference/README.md`.

USAGE
-----
    export MOLGPKA_SRC=~/MolGpKa/src
    python -m train.build_features --out data/features

Resumable: an array that already exists is skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_features")

FP_SPECS = (
    ("morgan_fps_2048.npy", "_MORGAN_FP"),
    ("atompair_fps_2048.npy", "_ATOMPAIR_FP"),
    ("toptorsion_fps_2048.npy", "_TT_2048_FP"),
)


def read_smiles(path: Path) -> list[str]:
    """SMILES indexed by row_idx, asserting the file really is 0..N-1 with no gaps."""
    rows: dict[int, str] = {}
    with path.open() as handle:
        for record in csv.DictReader(handle):
            rows[int(record["row_idx"])] = record["smiles"]
    expected = set(range(len(rows)))
    if set(rows) != expected:
        missing = sorted(expected - set(rows))[:5]
        raise ValueError(f"row_idx is not a dense 0..N-1 range; first missing: {missing}")
    return [rows[i] for i in range(len(rows))]


def build_fingerprints(smiles: list[str], out_dir: Path, force: bool) -> None:
    from rdkit import Chem, RDLogger

    from inference import featurize as fz

    RDLogger.DisableLog("rdApp.*")
    targets = [(name, getattr(fz, attr)) for name, attr in FP_SPECS]
    todo = [(n, g) for n, g in targets if force or not (out_dir / n).exists()]
    if not todo:
        log.info("fingerprints already built, skipping")
        return

    arrays = {name: np.zeros((len(smiles), 2048), dtype=np.float32) for name, _ in todo}
    failures = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures += 1
            continue
        for name, generator in todo:
            arrays[name][i] = fz._fp_array(generator, mol, 2048)
        if i and i % 25_000 == 0:
            log.info("  fingerprints %d/%d", i, len(smiles))
    for name, array in arrays.items():
        np.save(out_dir / name, array)
        log.info("  wrote %s %s", name, array.shape)
    if failures:
        log.warning("%d SMILES failed to parse and were left as zero rows", failures)


def build_descriptors(smiles: list[str], out_dir: Path, force: bool) -> None:
    """The 20-dim descriptor block, 7 columns of which need MolGpKa.

    Runs single-process: MolGpKa holds two GCN models per process, and forking them costs
    more memory than the loop saves in wall-clock at this size.
    """
    path = out_dir / "descriptors_20.npy"
    if path.exists() and not force:
        log.info("descriptors already built, skipping")
        return

    from inference.featurize import compute_descriptors

    out = np.zeros((len(smiles), 20), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i, smi in enumerate(smiles):
            out[i] = compute_descriptors(smi)
            if i and i % 10_000 == 0:
                log.info("  descriptors %d/%d", i, len(smiles))
        degraded = [w for w in caught if issubclass(w.category, RuntimeWarning)]

    np.save(path, out)
    log.info("  wrote %s %s", path.name, out.shape)
    if degraded:
        log.error(
            "MolGpKa DEGRADED: %s -- the 7 ionization descriptors are sentinels and this "
            "cache should be discarded once MolGpKa is fixed",
            str(degraded[0].message).split(".")[0],
        )


def build_chemberta(smiles: list[str], out_dir: Path, force: bool, batch: int) -> None:
    path = out_dir / "chemberta_384.npy"
    if path.exists() and not force:
        log.info("chemberta already built, skipping")
        return

    from model.chemberta_encoder import ChemBERTaEncoder

    encoder = ChemBERTaEncoder()
    out = np.zeros((len(smiles), 384), dtype=np.float32)
    for start in range(0, len(smiles), batch):
        chunk = smiles[start : start + batch]
        out[start : start + len(chunk)] = np.asarray(encoder.encode(chunk), dtype=np.float32)
        if start and start % (batch * 50) == 0:
            log.info("  chemberta %d/%d", start, len(smiles))
    np.save(path, out)
    log.info("  wrote %s %s", path.name, out.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, default=REPO / "data/compounds/compounds.csv")
    parser.add_argument("--out", type=Path, default=REPO / "data/features")
    parser.add_argument("--batch", type=int, default=256, help="ChemBERTa batch size")
    parser.add_argument("--force", action="store_true", help="rebuild arrays that exist")
    parser.add_argument(
        "--only",
        choices=["fingerprints", "descriptors", "chemberta"],
        help="build just one block",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    smiles = read_smiles(args.compounds)
    log.info("%d compounds from %s", len(smiles), args.compounds)

    if args.only in (None, "fingerprints"):
        build_fingerprints(smiles, args.out, args.force)
    if args.only in (None, "descriptors"):
        build_descriptors(smiles, args.out, args.force)
    if args.only in (None, "chemberta"):
        build_chemberta(smiles, args.out, args.force, args.batch)

    log.info("done -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
