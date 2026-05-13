"""Stage 1 training step -- the paper Methods recipe as one
function.

This is a *reference* train step, not a runnable end-to-end trainer:
loading the 7526-dim feature batches, building the NoamLR scheduler,
running validation + early stopping, and persisting the best checkpoint
are left to the user because they depend on the data layout you choose.

What is faithful to the paper:

  - Focal-BCE on each classification head with focal_gamma = 2.0 and a
    per-head pos_weight = n_negative / n_positive capped at 8 (paper).
  - MSE on each regression head, against training-fold-z-scored targets.
  - Per-head NaN masking so compounds without a label for that head are
    skipped (the labels matrix is sparse).
  - Mean over the active heads (uniform task weighting).
  - Adam optimiser with weight_decay = 1e-5; gradient clipping at L2 norm
    1.0 just before optimiser.step().
  - LR is whatever the caller's NoamLR scheduler has set on the optimiser
    for this step (paper: linear warmup 1e-4 -> 1e-3 over 3 epochs, then
    exponential decay to 1e-5 over the remainder).

What is *not* in this function: the SWA / "use final state" / V4-era
contrastive-cliff paths that lived in the deployed codebase as opt-in
internal modes. They are not part of the paper's recipe and are not
shipped here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from train.losses import focal_bce_with_logits

CLASSIFICATION_HEADS: tuple[str, ...] = (
    "herg_blocker_10um",
    "herg_blocker_1um",
    "nav15_blocker",
    "cav12_blocker",
    "iks_blocker",
)
REGRESSION_HEADS: tuple[str, ...] = ("herg_pchembl", "nav15_pchembl", "cav12_pchembl")


def stage1_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    head_to_col: dict[str, int],
    pos_weights: dict[str, float],
    reg_scalers: dict[int, tuple[float, float]],
    *,
    focal_gamma: float = 2.0,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    """One Stage 1 training step.

    Args
    ----
    model
        A `CrossAttnIonChannelPredictor` (or compatible) instance in train()
        mode.
    optimizer
        Pre-stepped externally; the function calls zero_grad / step.
    x
        (B, 7526) feature batch.
    y
        (B, 8) label matrix in column order
        [herg_pchembl, herg_10uM, herg_1uM, nav_pchembl, nav_block,
         cav_pchembl, cav_block, iks_block]. NaN where a compound has
        no label for that column.
    head_to_col
        Map from head name to column index into `y`.
    pos_weights
        Map from classification head name to its pos_weight (n_neg / n_pos,
        capped at 8). Computed once per training run from the full train
        fold and reused.
    reg_scalers
        Map from regression column index to (mu, sigma) computed on the
        train fold. Targets are z-scored before MSE.
    focal_gamma, grad_clip_norm
        Paper defaults.

    Returns
    -------
    dict of {"cls_loss", "reg_loss", "total_loss"} as Python floats.
    """
    optimizer.zero_grad()
    out = model(x)

    device = x.device
    cls_loss = torch.tensor(0.0, device=device)
    reg_loss = torch.tensor(0.0, device=device)
    n_cls = 0
    n_reg = 0

    for head in CLASSIFICATION_HEADS:
        yi = y[:, head_to_col[head]]
        mask = ~torch.isnan(yi)
        if mask.sum() == 0:
            continue
        per = focal_bce_with_logits(
            out[head][mask],
            yi[mask],
            focal_gamma=focal_gamma,
            pos_weight=pos_weights[head],
        )
        cls_loss = cls_loss + per.mean()
        n_cls += 1
    if n_cls:
        cls_loss = cls_loss / n_cls

    for head in REGRESSION_HEADS:
        col_i = head_to_col[head]
        yi = y[:, col_i]
        mask = ~torch.isnan(yi)
        if mask.sum() == 0:
            continue
        pred = out[head][mask]
        target = yi[mask]
        if col_i in reg_scalers:
            mu, sigma = reg_scalers[col_i]
            target = (target - mu) / max(sigma, 1e-6)
        reg_loss = reg_loss + (pred - target).pow(2).mean()
        n_reg += 1
    if n_reg:
        reg_loss = reg_loss / n_reg

    total = cls_loss + reg_loss
    total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()

    return {
        "cls_loss": float(cls_loss.detach()),
        "reg_loss": float(reg_loss.detach()),
        "total_loss": float(total.detach()),
    }
