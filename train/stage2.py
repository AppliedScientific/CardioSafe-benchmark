"""Stage 2 cliff fine-tune for CardioSafe (paper Methods section, paragraph
labelled "Stage 2: cliff fine-tune").

For each of the five seeds 42-46:
  - Load the Stage 1 checkpoint plus its descriptor and L1000
    scaler statistics; these are inherited unchanged.
  - For each of 9 epochs of 120 mini-batches:
      * Sample 512 ChEMBL training rows (random with replacement across
        epochs, without replacement within a mini-batch).
      * Concatenate the curated cliff rows (48 for tan70, 51 for tan60),
        each carrying an 8-fold sample weight on the focal-BCE loss.
      * Loss = focal-BCE on the five classification heads + MSE on the
        three regression heads + 0.3 * pairwise margin ranking loss on
        cliff pairs for the two hERG classification heads.
      * AdamW (lr 1e-5, weight_decay 1e-5), gradient clipping at L2 norm 1.0.
  - Save the fine-tuned checkpoint.

This script is the paper-faithful reference. It takes file-path arguments
for every input artefact so it can be wired to your own feature cache --
nothing in here is specific to the authors' infrastructure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

# Make the `model/` and `train/` sibling directories importable when this
# script is run as `python train/stage2.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.cross_attn import (  # noqa: E402
    CLASSIFICATION_HEADS,
    REGRESSION_HEADS,
    CrossAttnIonChannelPredictor,
)
from train.losses import focal_bce_with_logits, pairwise_margin_loss  # noqa: E402

# Label column order in the labels matrix.
COL_HERG_PCHEMBL = 0
COL_HERG_10UM = 1
COL_HERG_1UM = 2
COL_NAV_PCHEMBL = 3
COL_NAV_BLOCK = 4
COL_CAV_PCHEMBL = 5
COL_CAV_BLOCK = 6
COL_IKS_BLOCK = 7

HEAD_TO_COL: dict[str, int] = {
    "herg_pchembl": COL_HERG_PCHEMBL,
    "nav15_pchembl": COL_NAV_PCHEMBL,
    "cav12_pchembl": COL_CAV_PCHEMBL,
    "herg_blocker_10um": COL_HERG_10UM,
    "herg_blocker_1um": COL_HERG_1UM,
    "nav15_blocker": COL_NAV_BLOCK,
    "cav12_blocker": COL_CAV_BLOCK,
    "iks_blocker": COL_IKS_BLOCK,
}


def build_batch(
    idx: np.ndarray,
    morgan: np.ndarray,
    atompair: np.ndarray,
    toptors: np.ndarray,
    desc: np.ndarray,
    chemberta: np.ndarray,
    bio: np.ndarray,
    desc_mean: np.ndarray,
    desc_std: np.ndarray,
    bio_mean: np.ndarray,
    bio_std: np.ndarray,
) -> np.ndarray:
    """Assemble a (B, 7526) flat feature tensor matching the paper layout."""
    m = np.asarray(morgan[idx], dtype=np.float32)
    a = np.asarray(atompair[idx], dtype=np.float32)
    t = np.asarray(toptors[idx], dtype=np.float32)
    d = np.asarray(desc[idx], dtype=np.float32)
    d = (d - desc_mean) / np.where(desc_std < 1e-10, 1.0, desc_std)
    chem = np.concatenate([m, a, t, d], axis=1)
    cb = np.asarray(chemberta[idx], dtype=np.float32)
    b = np.asarray(bio[idx], dtype=np.float32)
    b = (b - bio_mean) / np.where(bio_std < 1e-10, 1.0, bio_std)
    return np.concatenate([chem, cb, b], axis=1)


def build_cliff_features(
    cliff: np.lib.npyio.NpzFile,
    desc_mean: np.ndarray,
    desc_std: np.ndarray,
    bio_mean: np.ndarray,
    bio_std: np.ndarray,
) -> np.ndarray:
    """Apply the same scaling to the precomputed cliff features.

    `cliff` is an NPZ with arrays:
        chem_raw   (N, 6164) -- [Morgan(2048) || AP(2048) || TT(2048) || desc(20)]
        chemberta  (N, 384)
        bio_raw    (N, 978)
    """
    chem_raw = cliff["chem_raw"].astype(np.float32)
    fp_block = chem_raw[:, :6144]
    desc_block = chem_raw[:, 6144:]
    desc_block = (desc_block - desc_mean) / np.where(desc_std < 1e-10, 1.0, desc_std)
    chem = np.concatenate([fp_block, desc_block], axis=1)
    cb = cliff["chemberta"].astype(np.float32)
    bio = cliff["bio_raw"].astype(np.float32)
    bio = (bio - bio_mean) / np.where(bio_std < 1e-10, 1.0, bio_std)
    return np.concatenate([chem, cb, bio], axis=1)


def build_pair_groups(
    pair_ids: list[str],
    roles: list[str],
) -> list[tuple[int, int]]:
    """Build all (blocker_index, safer_index) pairs from the cliff manifest."""
    groups: dict[str, dict[str, list[int]]] = {}
    for i, (pid, role) in enumerate(zip(pair_ids, roles, strict=True)):
        groups.setdefault(pid, {"blocker": [], "safer": []})[role].append(i)
    pairs: list[tuple[int, int]] = []
    for g in groups.values():
        for b in g["blocker"]:
            for s in g["safer"]:
                pairs.append((b, s))
    return pairs


def finetune_seed(seed: int, args: argparse.Namespace) -> None:
    morgan = np.load(args.morgan_path, mmap_mode="r")
    atompair = np.load(args.atompair_path, mmap_mode="r")
    toptors = np.load(args.toptors_path, mmap_mode="r")
    desc = np.load(args.descriptors_path, mmap_mode="r")
    chemberta = np.load(args.chemberta_path, mmap_mode="r")
    bio = np.load(args.bio_path, mmap_mode="r")
    labels = np.load(args.labels_path)
    splits = np.load(args.splits_path)
    train_idx = splits["train"]

    cliff = np.load(args.cliff_path, allow_pickle=True)

    base_dir = Path(args.stage1_dir) / f"seed_{seed}"
    ckpt = torch.load(base_dir / "model.pt", map_location="cpu", weights_only=False)
    config = json.loads((base_dir / "config.json").read_text())

    desc_mean = ckpt["descriptor_scaler_means"]
    desc_std = ckpt["descriptor_scaler_stds"]
    bio_mean = ckpt["l1000_norm_means"]
    bio_std = ckpt["l1000_norm_stds"]
    reg_scalers = ckpt["reg_scalers"]

    device = torch.device(args.device)
    model = CrossAttnIonChannelPredictor(
        chem_dim=config["chem_dim"],
        chemberta_dim=config["chemberta_dim"],
        bio_dim=config["bio_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    x_cliff_np = build_cliff_features(cliff, desc_mean, desc_std, bio_mean, bio_std)
    x_cliff = torch.from_numpy(x_cliff_np).to(device)
    y_cliff = torch.from_numpy(cliff["labels"].astype(np.float32)).to(device)
    n_cliff = x_cliff.shape[0]

    pair_ids = [str(p) for p in cliff["pair_ids"]]
    roles = [str(r) for r in cliff["roles"]]
    ranking_pairs = build_pair_groups(pair_ids, roles)
    print(f"  built {len(ranking_pairs)} (blocker, safer) ranking pairs")

    pos_weights: dict[str, float] = {}
    for head in CLASSIFICATION_HEADS:
        col = labels[train_idx, HEAD_TO_COL[head]]
        col = col[~np.isnan(col)]
        if len(col) == 0 or col.sum() == 0:
            pos_weights[head] = 1.0
            continue
        n_pos = col.sum()
        n_neg = len(col) - n_pos
        pw = n_neg / max(n_pos, 1)
        pos_weights[head] = float(min(pw, args.pos_weight_cap))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * args.steps_per_epoch

    model.train()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(total_steps):
        chembl_idx = rng.choice(train_idx, size=args.bs_chembl, replace=False)
        x_chembl_np = build_batch(
            chembl_idx,
            morgan,
            atompair,
            toptors,
            desc,
            chemberta,
            bio,
            desc_mean,
            desc_std,
            bio_mean,
            bio_std,
        )
        x_chembl = torch.from_numpy(x_chembl_np).to(device)
        y_chembl = torch.from_numpy(labels[chembl_idx].astype(np.float32)).to(device)

        x_all = torch.cat([x_chembl, x_cliff], dim=0)
        y_all = torch.cat([y_chembl, y_cliff], dim=0)
        sw_all = torch.cat(
            [
                torch.ones(args.bs_chembl, device=device),
                torch.full((n_cliff,), args.cliff_weight, device=device),
            ],
            dim=0,
        )

        optimizer.zero_grad()
        out = model(x_all)

        cls_loss = torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)
        n_cls = 0
        n_reg = 0
        for head in CLASSIFICATION_HEADS:
            col_i = HEAD_TO_COL[head]
            yi = y_all[:, col_i]
            mask = ~torch.isnan(yi)
            if mask.sum() == 0:
                continue
            per = focal_bce_with_logits(
                out[head][mask],
                yi[mask],
                focal_gamma=args.focal_gamma,
                pos_weight=pos_weights[head],
            )
            sw = sw_all[mask]
            cls_loss = cls_loss + (per * sw).sum() / sw.sum()
            n_cls += 1
        for head in REGRESSION_HEADS:
            col_i = HEAD_TO_COL[head]
            yi = y_all[:, col_i]
            mask = ~torch.isnan(yi)
            if mask.sum() == 0:
                continue
            pred = out[head][mask]
            target = yi[mask]
            if col_i in reg_scalers:
                mu, sigma = reg_scalers[col_i]
                target = (target - mu) / max(sigma, 1e-6)
            mse = (pred - target).pow(2)
            sw = sw_all[mask]
            reg_loss = reg_loss + (mse * sw).sum() / sw.sum()
            n_reg += 1
        if n_cls:
            cls_loss = cls_loss / n_cls
        if n_reg:
            reg_loss = reg_loss / n_reg

        rank_loss = torch.tensor(0.0, device=device)
        cliff_offset = args.bs_chembl
        for head, col_i in (
            ("herg_blocker_1um", COL_HERG_1UM),
            ("herg_blocker_10um", COL_HERG_10UM),
        ):
            head_rank = torch.tensor(0.0, device=device)
            for bi, si in ranking_pairs:
                yb = y_cliff[bi, col_i]
                ys = y_cliff[si, col_i]
                if torch.isnan(yb) or torch.isnan(ys) or yb <= ys:
                    continue
                lb = out[head][cliff_offset + bi]
                ls = out[head][cliff_offset + si]
                head_rank = head_rank + pairwise_margin_loss(lb, ls, margin=args.pair_margin)
            head_rank = head_rank / max(len(ranking_pairs), 1)
            rank_loss = rank_loss + head_rank

        total_loss = cls_loss + reg_loss + args.pair_lambda * rank_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == total_steps - 1:
            dt = time.time() - t0
            print(
                f"  seed={seed} step {step + 1:4d}/{total_steps}  "
                f"cls={cls_loss.item():.4f}  reg={reg_loss.item():.4f}  "
                f"rank={rank_loss.item():.4f}  total={total_loss.item():.4f}  [{dt:.0f}s]",
                flush=True,
            )

    out_dir = Path(args.output_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "descriptor_scaler_means": desc_mean,
            "descriptor_scaler_stds": desc_std,
            "reg_scalers": reg_scalers,
            "l1000_norm_means": bio_mean,
            "l1000_norm_stds": bio_std,
        },
        out_dir / "model.pt",
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  saved seed {seed} -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=9, help="Paper recipe: 9.")
    p.add_argument("--steps-per-epoch", type=int, default=120, help="Paper recipe: 120.")
    p.add_argument("--bs-chembl", type=int, default=512, help="Paper recipe: 512.")
    p.add_argument("--cliff-weight", type=float, default=8.0, help="Paper recipe: 8.0.")
    p.add_argument("--pair-lambda", type=float, default=0.3, help="Paper recipe: 0.3.")
    p.add_argument("--pair-margin", type=float, default=1.5, help="Paper recipe: 1.5.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Paper recipe: 2.0.")
    p.add_argument("--pos-weight-cap", type=float, default=8.0, help="Paper recipe: 8.")
    p.add_argument("--lr", type=float, default=1e-5, help="Paper recipe: 1e-5.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    p.add_argument("--stage1-dir", type=Path, required=True,
                   help="Directory containing Stage 1 checkpoints seed_42/.. seed_46/.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for fine-tuned checkpoints.")
    p.add_argument("--morgan-path", type=Path, required=True)
    p.add_argument("--atompair-path", type=Path, required=True)
    p.add_argument("--toptors-path", type=Path, required=True)
    p.add_argument("--descriptors-path", type=Path, required=True)
    p.add_argument("--chemberta-path", type=Path, required=True)
    p.add_argument("--bio-path", type=Path, required=True,
                   help="Per-compound L1000 predicted z-scores (N, 978).")
    p.add_argument("--labels-path", type=Path, required=True)
    p.add_argument("--splits-path", type=Path, required=True,
                   help="NPZ with train/val/test row_idx arrays.")
    p.add_argument("--cliff-path", type=Path, required=True,
                   help="NPZ with chem_raw / chemberta / bio_raw / labels / pair_ids / roles.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"\n{'=' * 72}\n  Stage 2 cliff fine-tune, seed {seed}\n{'=' * 72}")
        finetune_seed(seed, args)


if __name__ == "__main__":
    main()
