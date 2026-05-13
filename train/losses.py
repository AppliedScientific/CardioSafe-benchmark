"""Loss functions used by CardioSafe training.

Both losses are stated in the Methods section of Jovanovic et al. 2026:

* `focal_bce_with_logits` -- focal-BCE for the five binary classification
  heads, per paper "focal loss (gamma = 2.0) with a per-head positive-class
  weight pos_weight = n_negative / n_positive, capped at 8."

* `pairwise_margin_loss` -- ranking loss applied to the two hERG
  classification heads on cliff pairs (Stage 2 fine-tune), per paper
  "max(0, 1.5 - (logit_blocker - logit_safer)) with coefficient 0.3."

Regression heads use plain MSE on training-fold-z-scored targets; that
is one line of PyTorch and is not wrapped here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    focal_gamma: float = 2.0,
    pos_weight: float = 1.0,
) -> Tensor:
    """Per-sample focal binary cross-entropy.

    loss_i = - pw * y * (1 - p)^gamma * log p   - (1 - y) * p^gamma * log(1 - p)

    where p = sigmoid(logits). Returns a (N,) tensor; the caller decides how
    to reduce (mean, weighted-mean, etc.) since cliff samples carry the
    Stage 2 8-fold weight via an external per-sample weight vector.
    """
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    p = torch.sigmoid(logits)
    loss_pos = -pos_weight * targets * (1.0 - p).pow(focal_gamma) * log_p
    loss_neg = -(1.0 - targets) * p.pow(focal_gamma) * log_1mp
    return loss_pos + loss_neg


def pairwise_margin_loss(
    logit_blocker: Tensor,
    logit_safer: Tensor,
    *,
    margin: float = 1.5,
) -> Tensor:
    """Margin ranking loss for cliff pairs.

    loss = max(0, margin - (logit_blocker - logit_safer))

    Returns a per-pair tensor; caller decides reduction.
    """
    return F.relu(margin - (logit_blocker - logit_safer))
