"""CardioSafe inference CLI.

Reads a CSV with a `smiles` column, scores it with the chosen ensemble
(v1.0 = paper preprint version; v1.1 = audit-clean retrain), and writes a
CSV with 8 columns of predictions appended.

Examples:
    python -m inference.predict --input molecules.csv --output scored.csv --version v1.1
    python -m inference.predict --input molecules.csv --output scored.csv --version v1.0 --device cpu

Weight files (~80 MB per version + 10 MB L1000 encoder) are downloaded from
the CardioSafe-benchmark GitHub Releases on first use and cached under
`~/.cache/cardiosafe/weights/` (override with $CARDIOSAFE_CACHE).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from model.chemberta_encoder import ChemBERTaEncoder
from model.cross_attn import ALL_HEADS

from .ensemble import load_ensemble, load_l1000_encoder, predict
from .featurize import featurize_batch


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> int:
    p = argparse.ArgumentParser(prog="cardiosafe-predict", description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="CSV with a 'smiles' column")
    p.add_argument("--output", required=True, type=Path, help="Where to write the scored CSV")
    p.add_argument("--version", choices=("v1.0", "v1.1"), default="v1.1",
                   help="Which ensemble to use (v1.1 = audit-clean retrain; default)")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column in --input")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        sys.exit(f"input CSV is missing the '{args.smiles_col}' column")
    smiles = df[args.smiles_col].astype(str).tolist()
    print(f"Loaded {len(smiles)} SMILES from {args.input}")

    print(f"Loading ChemBERTa encoder...")
    chemberta = ChemBERTaEncoder()
    print(f"Loading L1000 encoder...")
    l1000 = load_l1000_encoder(device=device)
    print(f"Loading CardioSafe ensemble ({args.version}, 5 seeds)...")
    ensemble = load_ensemble(args.version, device=device)

    all_results = {h: [] for h in ALL_HEADS}
    for start in range(0, len(smiles), args.batch_size):
        end = min(start + args.batch_size, len(smiles))
        batch_smiles = smiles[start:end]
        print(f"  scoring {start+1}..{end} of {len(smiles)}")
        batch = featurize_batch(batch_smiles, chemberta_encoder=chemberta, l1000_encoder=l1000)
        preds = predict(batch, ensemble=ensemble, device=device)
        for h in ALL_HEADS:
            all_results[h].extend(preds[h].tolist())

    for h in ALL_HEADS:
        df[f"cardiosafe_{args.version}_{h}"] = all_results[h]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
