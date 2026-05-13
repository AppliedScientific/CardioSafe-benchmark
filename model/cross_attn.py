"""CardioSafe cross-attention architecture.

Reference implementation of the network described in the Methods section
of Jovanovic et al. 2026 ("CardioSafe: multi-task prediction of cardiac
ion channel activity with reverse-leak audited benchmarking"). Forward
pass only; no losses, no training loop, no weights.

Three modality branches feed a per-channel cross-attention head:

    Branch A (chem fingerprints + descriptors, 6164 dim)
        Linear(6164, 512) -> BN -> ReLU -> Dropout(0.2)
        Linear(512, 256) -> BN -> ReLU -> Dropout(0.2)
    Branch B (ChemBERTa-77M-MTR sentence embedding, 384 dim)
        Linear(384, 128) -> LayerNorm -> ReLU -> Dropout(0.2)
    Branch C (L1000 predicted gene expression, 978 dim)
        Linear(978, 128) -> LayerNorm -> ReLU -> Dropout(0.3)

    Cross-attention: 4 learnable channel queries x 3 modality K/V tokens.
        attn_dim = 128, n_heads = 2, attention dropout = 0.1
        Query init: torch.randn(4, 128) * 0.02

    Per-channel head: [attn_context (128) || chem_residual (256)] -> 384
        Linear(384, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1)

Input layout (single flat tensor):
    x[:, :6164]              chem features (Morgan || AtomPair || TT || desc20)
    x[:, 6164:6164+384]      ChemBERTa embedding
    x[:, 6164+384:]          L1000 predicted z-scores (978)

Outputs (dict by head name):
    Regression heads (raw pChEMBL):
        herg_pchembl, nav15_pchembl, cav12_pchembl
    Classification heads (raw logits):
        herg_blocker_10um, herg_blocker_1um, nav15_blocker, cav12_blocker, iks_blocker

Channel ordering for cross-attention queries:
    0: hERG    1: Nav1.5    2: Cav1.2    3: IKs

Weight init: Kaiming-normal (ReLU nonlinearity) for every Linear layer,
zero biases. The four channel queries use the small-scale Gaussian init
above. Seepaper Methods for the full specification.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

REGRESSION_HEADS: tuple[str, ...] = ("herg_pchembl", "nav15_pchembl", "cav12_pchembl")
CLASSIFICATION_HEADS: tuple[str, ...] = (
    "herg_blocker_10um",
    "herg_blocker_1um",
    "nav15_blocker",
    "cav12_blocker",
    "iks_blocker",
)
ALL_HEADS: tuple[str, ...] = REGRESSION_HEADS + CLASSIFICATION_HEADS

HEAD_TO_CHANNEL: dict[str, int] = {
    "herg_pchembl": 0,
    "herg_blocker_10um": 0,
    "herg_blocker_1um": 0,
    "nav15_pchembl": 1,
    "nav15_blocker": 1,
    "cav12_pchembl": 2,
    "cav12_blocker": 2,
    "iks_blocker": 3,
}

CHEM_DIM = 6164
CHEMBERTA_DIM = 384
BIO_DIM = 978


def _build_heads(in_dim: int, head_hidden: int, dropout: float) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            name: nn.Sequential(
                nn.Linear(in_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for name in ALL_HEADS
        }
    )


def _init_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class CrossAttnIonChannelPredictor(nn.Module):
    """Per-channel cross-attention routing over three modality branches.

    Inputs:
        x: (B, 7526) float tensor. Layout: [chem (6164) || chemberta (384) || bio (978)].

    Outputs:
        dict mapping head name (8 entries: 3 regression + 5 classification) to
        a (B,) tensor. Regression heads return raw pChEMBL predictions;
        classification heads return raw logits (apply sigmoid for probabilities).
    """

    REGRESSION_HEADS = REGRESSION_HEADS
    CLASSIFICATION_HEADS = CLASSIFICATION_HEADS
    HEAD_TO_CHANNEL = HEAD_TO_CHANNEL
    N_CHANNELS = 4

    def __init__(
        self,
        chem_dim: int = CHEM_DIM,
        chemberta_dim: int = CHEMBERTA_DIM,
        bio_dim: int = BIO_DIM,
        chem_hidden: int = 512,
        chem_out: int = 256,
        attn_dim: int = 128,
        n_heads: int = 2,
        attn_dropout: float = 0.1,
        head_hidden: int = 128,
        chem_dropout: float = 0.2,
        chemberta_dropout: float = 0.2,
        bio_dropout: float = 0.3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.chem_dim = chem_dim
        self.chemberta_dim = chemberta_dim
        self.bio_dim = bio_dim
        self.chem_out = chem_out
        self.attn_dim = attn_dim

        # Branch A: chem fingerprints + descriptors -- BN, low dropout.
        self.chem_branch = nn.Sequential(
            nn.Linear(chem_dim, chem_hidden),
            nn.BatchNorm1d(chem_hidden),
            nn.ReLU(),
            nn.Dropout(chem_dropout),
            nn.Linear(chem_hidden, chem_out),
            nn.BatchNorm1d(chem_out),
            nn.ReLU(),
            nn.Dropout(chem_dropout),
        )

        # Branch B: ChemBERTa -- single-layer projection with LayerNorm.
        self.chemberta_branch = nn.Sequential(
            nn.Linear(chemberta_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(chemberta_dropout),
        )

        # Branch C: L1000 predicted gene expression -- single-layer with LayerNorm.
        self.bio_branch = nn.Sequential(
            nn.Linear(bio_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(bio_dropout),
        )

        # K/V projections.
        self.chem_proj = nn.Sequential(
            nn.Linear(chem_out, attn_dim),
            nn.LayerNorm(attn_dim),
        )
        self.cb_proj = nn.LayerNorm(attn_dim)
        self.bio_proj = nn.LayerNorm(attn_dim)

        # Per-channel learnable queries.
        self.queries = nn.Parameter(torch.randn(self.N_CHANNELS, attn_dim) * 0.02)

        # Cross-attention.
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Per-channel heads: [attn_context || chem_residual].
        head_in_dim = attn_dim + chem_out
        self.heads = _build_heads(head_in_dim, head_hidden, dropout)
        _init_weights(self)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        chem_x = x[:, : self.chem_dim]
        cb_x = x[:, self.chem_dim : self.chem_dim + self.chemberta_dim]
        bio_x = x[:, self.chem_dim + self.chemberta_dim :]

        chem_e = self.chem_branch(chem_x)            # (B, chem_out=256)
        cb_e = self.chemberta_branch(cb_x)           # (B, attn_dim=128)
        bio_e = self.bio_branch(bio_x)               # (B, attn_dim=128)

        chem_kv = self.chem_proj(chem_e)             # (B, attn_dim)
        cb_kv = self.cb_proj(cb_e)                   # (B, attn_dim)
        bio_kv = self.bio_proj(bio_e)                # (B, attn_dim)
        kv = torch.stack([chem_kv, cb_kv, bio_kv], dim=1)  # (B, 3, attn_dim)

        batch = kv.size(0)
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)  # (B, 4, attn_dim)

        ctx, _ = self.attn(q, kv, kv)                # (B, 4, attn_dim)

        out: dict[str, Tensor] = {}
        for name in ALL_HEADS:
            ch = self.HEAD_TO_CHANNEL[name]
            head_in = torch.cat([ctx[:, ch, :], chem_e], dim=1)
            out[name] = self.heads[name](head_in).squeeze(-1)
        return out
