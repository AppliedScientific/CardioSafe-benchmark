"""The CardioSafe architecture: three branches, seven heads.

Successor to the cross-attention model in `cross_attn.py`, which stays in this repo
because the published v1.0 / v1.1 ensemble weights load into it. This class provides the
plain-concatenation reference and several historical fusion variants. The recorded
comparisons did not establish an adaptive-fusion advantage. Weight variation alone
does not establish whether routing is constant or useful; see reports/phase4_findings.md
for corrected diagnostics and controlled prediction interventions.

Two further departures from the internal arm this was developed as:

  NO BIOLOGICAL BRANCH. That arm carried a 117-column block -- 21 read-across statistics
  over nearest training neighbours plus 96 gnina docking features -- routed per channel
  into every head. It is removed here rather than zeroed. Those are not the same
  experiment: a branch fed zeros still spends parameters and still lets its BatchNorm
  move, so an ablation that keeps the module measures something other than the model
  without it. Heads now see chemistry alone.

  RAW FINGERPRINTS. That arm used Sort-and-Slice Morgan over hashed AtomPair and
  TopTorsion. Sort-and-Slice ranks substructures by their support in a split's TRAIN
  fold, so the array cannot be rebuilt from released data without also fixing the split
  it was ranked on. Plain hash-folded 2048-bit fingerprints are reproducible from SMILES
  alone, which is what a public benchmark needs.

Both changes mean numbers from this class need their own baseline rather than being read
against any previously recorded result.

Input layout is one flat tensor:

    x = [ chem_x (chem_dim) | chemberta (chemberta_dim) ]

where chem_x is [ fingerprints (chem_dim - n_desc) | descriptors (n_desc) ]. With three
2048-bit fingerprints and 20 descriptors that is 6144 + 20 + 384 = 6548 columns.

Every head sees [fp_e 256 || desc_e 64 || cb_e 128] = 448.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def _init_weights(model: nn.Module) -> None:
    """Kaiming-normal on every Linear, zero bias.

    PyTorch's default Linear init is kaiming-UNIFORM with a uniform bias, so a run started
    from the default would not be comparable to any previously measured number.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


REGRESSION_HEADS = ("herg_pchembl", "nav15_pchembl", "cav12_pchembl")
CLASSIFICATION_HEADS = (
    "herg_blocker_10um",
    "herg_blocker_1um",
    "nav15_blocker",
    "cav12_blocker",
)
ALL_HEADS = REGRESSION_HEADS + CLASSIFICATION_HEADS

# Ion channel behind each head. Both hERG binary screens and the hERG regression head share
# one channel: they are the same site measured at different cutoffs. Only the attention
# variants read this -- with `attention="none"` every head sees the same trunk.
CHANNELS = ("herg", "nav15", "cav12")
HEAD_CHANNEL = {
    "herg_pchembl": "herg",
    "herg_blocker_10um": "herg",
    "herg_blocker_1um": "herg",
    "nav15_pchembl": "nav15",
    "nav15_blocker": "nav15",
    "cav12_pchembl": "cav12",
    "cav12_blocker": "cav12",
}

# Nominal pChEMBL boundary behind each binary label, measured on the 18,051 compounds that
# carry both an exact value and a call: Youden puts hERG's 10 uM screen at 5.006 with 95.6%
# agreement at exactly 5.0, and the 1 uM screen at 6.000 with 98.8%. This model does not use
# these -- its classification heads learn their own boundary from the labels. They are here
# so anything reading a prediction knows what the binary labels meant, and so a consistency
# check between the two output families has a number to check against.
NOMINAL_TAU = {
    "herg_blocker_10um": 5.0,
    "herg_blocker_1um": 6.0,
    "nav15_blocker": 5.0,
    "cav12_blocker": 5.0,
}


# Descriptor sub-blocks, from hierarchical clustering of the 20 descriptors on the train fold
# (correlation distance, average linkage) rather than from intuition. The grouping is
# chemically coherent and recovers the canonical hERG pharmacophore: ionization dominates
# (frac_cation univariate AUC 0.797, pka_basic 0.783), lipophilicity and bulk add (clogP
# 0.687, heavy atoms 0.697), and polarity protects (TPSA 0.388 and frac_anion 0.277, both
# BELOW chance).
#
# Keeping them apart matters because they measure genuinely different things and a molecule
# can be informative on one and silent on another -- 16.8% of the pool has no detectable
# basic centre, which makes the whole ionization block constant for those compounds.
#
# ELECTROSTATICS is retained as its own block despite being nearly inert (four of its five
# descriptors sit within 0.07 of chance, including max_pos_n at 0.487 -- the descriptor added
# specifically for the basic-nitrogen pharmacophore, which frac_cation captures far better).
# Isolating it means a gate can learn to close it, which is a cleaner statement than burying
# it among descriptors that work.
DESCRIPTOR_BLOCKS = {
    "ionization": (13, 14, 16, 17, 18, 19),   # pka_acidic, pka_basic, frac cat/ani/zwit/neut
    "lipophilicity": (1, 6, 15),              # clogP, aromatic rings, logD 7.4
    "size_polarity": (0, 2, 3, 4, 5, 7),      # MW, TPSA, HBD, HBA, rot bonds, heavy atoms
    "electrostatics": (8, 9, 10, 11, 12),     # Gasteiger mean/max/min/std, max+ on N
}

# Buffers this class owns that no checkpoint predating it carries.
OWN_BUFFERS = ("reg_mean", "reg_std")


class CardioSafe(nn.Module):
    """Three branches into seven heads.

    Args:
        x: (B, chem_dim + chemberta_dim) float tensor, laid out as documented above.
    """

    def __init__(
        self,
        chem_dim: int = 6164,
        chemberta_dim: int = 384,
        n_desc: int = 20,
        chem_hidden: int = 512,
        chem_out: int = 256,
        desc_hidden: int = 64,
        branch_dim: int = 128,
        head_hidden: int = 128,
        chem_dropout: float = 0.2,
        desc_dropout: float = 0.1,
        chemberta_dropout: float = 0.2,
        dropout: float = 0.3,
        chemberta: bool = True,
        attention: str = "none",
        attn_dim: int = 128,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if attention not in ("none", "static", "trunk", "centered", "gated"):
            raise ValueError(
                "attention must be none|static|trunk|centered|gated, got "
                f"{attention!r}"
            )
        self.chem_dim = chem_dim
        self.chemberta_dim = chemberta_dim
        self.n_desc = n_desc
        self.fp_dim = chem_dim - n_desc
        self.attention = attention
        self.use_chemberta = chemberta

        # BatchNorm on the fingerprint block because its per-feature scale is a property of
        # the dataset, not of a molecule; LayerNorm on ChemBERTa because that has a
        # meaningful per-molecule distribution a batch statistic would flatten.
        self.fp_branch = nn.Sequential(
            nn.Linear(self.fp_dim, chem_hidden),
            nn.BatchNorm1d(chem_hidden),
            nn.ReLU(),
            nn.Dropout(chem_dropout),
            nn.Linear(chem_hidden, chem_out),
            nn.BatchNorm1d(chem_out),
            nn.ReLU(),
            nn.Dropout(chem_dropout),
        )

        # BatchNorm here too, and it is the reason the old descriptor bypass could be dropped
        # rather than kept alongside. LayerNorm would normalise across the 64 hidden units
        # per molecule, which discards the absolute magnitude of the representation -- and
        # absolute magnitude is exactly what a descriptor asserts. "Strongly basic and very
        # lipophilic" is a claim about where this compound sits on a physical scale relative
        # to other compounds, and BatchNorm's dataset-level per-feature statistics preserve
        # that ordering where LayerNorm renormalises it away.
        #
        # Retain the baseline's two nonlinear descriptor blocks. Layer count does not
        # directly specify an interaction order; even one hidden layer can represent
        # interactions involving many input features.
        self.desc_branch = nn.Sequential(
            nn.Linear(n_desc, desc_hidden),
            nn.BatchNorm1d(desc_hidden),
            nn.ReLU(),
            nn.Dropout(desc_dropout),
            nn.Linear(desc_hidden, desc_hidden),
            nn.BatchNorm1d(desc_hidden),
            nn.ReLU(),
            nn.Dropout(desc_dropout),
        )

        # The one branch that keeps LayerNorm -- a transformer embedding's meaning is its
        # direction, which is why cosine is how such things are compared.
        if chemberta:
            self.chemberta_branch = nn.Sequential(
                nn.Linear(chemberta_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.ReLU(),
                nn.Dropout(chemberta_dropout),
            )

        # The regression heads are trained against STANDARDISED targets, so a raw head output
        # is a z-score, not a pChEMBL value. A caller who reads the head directly is off by
        # roughly a factor of 5.4 in the mean and 1.6 in the spread. These default to identity
        # so a freshly constructed model is honest about not having them; load_checkpoint
        # fills them in.
        self.register_buffer("reg_mean", torch.zeros(len(REGRESSION_HEADS)))
        self.register_buffer("reg_std", torch.ones(len(REGRESSION_HEADS)))

        if attention != "gated":
            self.shared_dim = chem_out + desc_hidden + (branch_dim if chemberta else 0)

        # GATED FUSION -- deliberately NOT attention pooling, and the distinction is the
        # point. Attention computes a convex combination, y = sum_i w_i v_i. That is the right
        # operator when tokens are SUBSTITUTABLE: transformer tokens are word embeddings in one
        # shared space, so averaging meanings is coherent. These blocks are COMPLEMENTARY, not
        # substitutable -- averaging "bulky and polar" with "carries a protonated basic
        # nitrogen" destroys both statements. That is why plain concatenation kept matching the
        # attention arms: concatenation preserves complementarity and pooling averages it away.
        #
        # So keep every block whole and modulate how much each contributes, per molecule and
        # per channel. Each block gets a scalar gate g in (0, 2), and the head sees the same
        # concatenation it always did with each slice scaled by its gate.
        #
        # The hypothesis this encodes is specific and falsifiable: 16.8% of the pool has no
        # detectable basic centre, which makes the entire ionization block constant for those
        # compounds, so its gate should close on them and lipophilicity should carry more. A
        # plain MLP can express that too -- ReLU can learn "if frac_cation is 0, ignore these
        # dims" -- so the gate makes it easier rather than possible, and the expected gain is
        # correspondingly modest.
        #
        # Gates initialise to EXACTLY 1.0 (zero weights, zero bias, g = 2*sigmoid(0)), so an
        # untrained gated model is bit-identical to the concat baseline and any difference is
        # something training chose.
        if attention == "gated":
            self.block_names = ["fingerprints"] + list(DESCRIPTOR_BLOCKS)
            if chemberta:
                self.block_names.append("chemberta")
            self.desc_branches = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(len(cols), 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(desc_dropout),
                        nn.Linear(32, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(desc_dropout),
                    )
                    for name, cols in DESCRIPTOR_BLOCKS.items()
                }
            )
            for name, cols in DESCRIPTOR_BLOCKS.items():
                self.register_buffer(
                    f"desc_cols_{name}", torch.tensor(cols, dtype=torch.long), persistent=False
                )
            widths = [chem_out] + [32] * len(DESCRIPTOR_BLOCKS) + ([branch_dim] if chemberta else [])
            self.block_widths = widths
            self.shared_dim = sum(widths)
            self.gate_net = nn.Linear(self.shared_dim, len(CHANNELS) * len(widths))
            self.head_in_dim = self.shared_dim
        else:
            self.block_names = []

        # Historical fusion variants: fixed channel queries (static), an added
        # molecule-conditioned query component (trunk), and optional logit BN
        # (centered). Fixed queries can still give adaptive weights through keys;
        # normalized key directions still vary with the molecule.
        #
        # Separate linear value projections retain branch magnitude, and the head
        # receives both the full concatenated trunk and the pooled context.
        # Three tokens do not mathematically force nearly constant attention.
        # The historical attn_heads argument is retained for API compatibility;
        # this implementation uses one cosine query per channel, not multi-head attention.
        if attention not in ("none", "gated"):
            token_dims = (chem_out, desc_hidden) + ((branch_dim,) if chemberta else ())
            self.key_proj = nn.ModuleList(
                [nn.Sequential(nn.Linear(d, attn_dim), nn.LayerNorm(attn_dim)) for d in token_dims]
            )
            self.value_proj = nn.ModuleList([nn.Linear(d, attn_dim) for d in token_dims])
            self.query_embed = nn.Parameter(torch.randn(len(CHANNELS), attn_dim) * 0.02)
            if attention in ("trunk", "centered"):
                self.query_from_trunk = nn.Linear(self.shared_dim, attn_dim)
            # Normalize query/key directions and learn the cosine-score temperature.
            # Scores are bounded by that temperature before optional logit BatchNorm.
            self.log_temp = nn.Parameter(torch.tensor(float(np.log(4.0))))
            if attention == "centered":
                # Standardize each (channel, token) logit with training-batch statistics.
                # Zero-mean logits do not force a uniform mean softmax. Calibration of
                # these running statistics alone can substantially increase weight SD,
                # even with every parameter frozen. Scalar p(1-p) heuristics also omit
                # cross-token terms in the full softmax Jacobian. See the Phase-4 audits.
                self.logit_norm = nn.BatchNorm1d(len(CHANNELS) * len(token_dims), affine=False)
            self.head_in_dim = self.shared_dim + attn_dim
        elif attention == "none":
            self.head_in_dim = self.shared_dim

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.head_in_dim, head_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
                for name in ALL_HEADS
            }
        )
        _init_weights(self)
        if attention == "gated":
            # After _init_weights, which would otherwise Kaiming-initialise the gate layer
            # and start the model at a random emphasis. Zeroed here so every gate is
            # exactly 2*sigmoid(0) = 1.0 and an untrained gated model is numerically the
            # concat baseline -- any difference is then attributable to training.
            nn.init.zeros_(self.gate_net.weight)
            nn.init.zeros_(self.gate_net.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.forward_with_embedding(x)[0]

    def forward_with_embedding(self, x: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        """Head outputs plus the chemistry embedding the cliff-pair contrastive loss uses.

        That embedding is [fp_e || desc_e], 320 wide rather than 256. Activity cliffs are
        pairs that a fingerprint calls nearly identical and an assay does not, so excluding
        the descriptor half would hand the contrastive loss exactly the representation that
        cannot separate the pair.
        """
        chem_x = x[:, : self.chem_dim]
        cb_x = x[:, self.chem_dim : self.chem_dim + self.chemberta_dim]

        chem_e = torch.cat(
            [self.fp_branch(chem_x[:, : self.fp_dim]), self.desc_branch(chem_x[:, self.fp_dim :])],
            dim=1,
        )
        if self.attention == "gated":
            blocks = self._blocks(chem_x, cb_x)
            gates = self._gates(torch.cat(blocks, dim=1))
            out = {
                name: head(
                    self._apply_gates(blocks, gates[:, CHANNELS.index(HEAD_CHANNEL[name])])
                ).squeeze(-1)
                for name, head in self.heads.items()
            }
            return out, chem_e

        branches = [self.fp_branch(chem_x[:, : self.fp_dim]), self.desc_branch(chem_x[:, self.fp_dim :])]
        if self.use_chemberta:
            branches.append(self.chemberta_branch(cb_x))
        branches = tuple(branches)
        shared = torch.cat(branches, dim=1)
        if self.attention == "none":
            out = {name: head(shared).squeeze(-1) for name, head in self.heads.items()}
            return out, chem_e

        context, _ = self._attend(branches, shared)
        out = {
            name: head(
                torch.cat([shared, context[:, CHANNELS.index(HEAD_CHANNEL[name])]], dim=1)
            ).squeeze(-1)
            for name, head in self.heads.items()
        }
        return out, chem_e

    def _attend(self, branches, shared: Tensor) -> tuple[Tensor, Tensor]:
        """(context per channel, attention weights). Weights are returned for diagnostics."""
        keys = torch.stack([p(b) for p, b in zip(self.key_proj, branches)], dim=1)
        values = torch.stack([p(b) for p, b in zip(self.value_proj, branches)], dim=1)
        query = self.query_embed.unsqueeze(0).expand(shared.shape[0], -1, -1)
        if self.attention in ("trunk", "centered"):
            query = query + self.query_from_trunk(shared).unsqueeze(1)
        q_hat = nn.functional.normalize(query, dim=-1)
        k_hat = nn.functional.normalize(keys, dim=-1)
        # clamped so a diverging temperature cannot silently turn this into a hard argmax
        temp = self.log_temp.clamp(-2.0, 4.0).exp()
        logits = torch.einsum("bcd,btd->bct", q_hat, k_hat)
        if self.attention == "centered":
            b, c, t_ = logits.shape
            logits = self.logit_norm(logits.reshape(b, c * t_)).reshape(b, c, t_)
        weights = (logits * temp).softmax(dim=-1)
        return torch.einsum("bct,btd->bcd", weights, values), weights

    @torch.no_grad()
    def attention_stats(self, x: Tensor) -> dict[str, float]:
        """Descriptive weight statistics, not a test of useful learned routing.

        Compare identical probes at initialization and after training, accounting for
        normalization-state changes. Small weight SD does not imply constant output:
        the values can still depend on the molecule. Predictive utility needs matched
        controls and route interventions, not an inference from SD or entropy alone.
        """
        if self.attention not in ("static", "trunk", "centered"):
            return {}
        chem_x = x[:, : self.chem_dim]
        cb_x = x[:, self.chem_dim : self.chem_dim + self.chemberta_dim]
        branches = [
            self.fp_branch(chem_x[:, : self.fp_dim]),
            self.desc_branch(chem_x[:, self.fp_dim :]),
        ]
        if self.use_chemberta:
            branches.append(self.chemberta_branch(cb_x))
        branches = tuple(branches)
        _, weights = self._attend(branches, torch.cat(branches, dim=1))
        entropy = -(weights.clamp_min(1e-9).log() * weights).sum(-1)
        return {
            "per_molecule_sd": float(weights.std(dim=0).mean()),
            "entropy": float(entropy.mean()),
            "entropy_max": float(np.log(weights.shape[-1])),
            "mean_weights": [round(float(v), 4) for v in weights.mean(dim=(0, 1))],
        }

    def _blocks(self, chem_x: Tensor, cb_x: Tensor) -> list[Tensor]:
        """One embedding per complementary block, kept separate rather than pooled."""
        desc_x = chem_x[:, self.fp_dim :]
        out = [self.fp_branch(chem_x[:, : self.fp_dim])]
        for name in DESCRIPTOR_BLOCKS:
            out.append(self.desc_branches[name](desc_x[:, getattr(self, f"desc_cols_{name}")]))
        if self.use_chemberta:
            out.append(self.chemberta_branch(cb_x))
        return out

    def _gates(self, trunk: Tensor) -> Tensor:
        """(batch, channels, blocks) in (0, 2), exactly 1.0 at initialisation."""
        raw = self.gate_net(trunk)
        return 2.0 * torch.sigmoid(raw).reshape(-1, len(CHANNELS), len(self.block_widths))

    def _apply_gates(self, blocks: list[Tensor], gate_row: Tensor) -> Tensor:
        """Scale each block by its gate and concatenate. Nothing is averaged away."""
        return torch.cat([b * gate_row[:, i : i + 1] for i, b in enumerate(blocks)], dim=1)

    @torch.no_grad()
    def gate_stats(self, x: Tensor) -> dict:
        """Descriptive gate variation; not identifiable chemical importance.

        Even normalized cross-block ratios can change under compensating head weights.
        Compare initial/trained behavior and prediction interventions before attributing
        predictive utility to molecule-dependent gating.
        """
        if self.attention != "gated":
            return {}
        chem_x = x[:, : self.chem_dim]
        cb_x = x[:, self.chem_dim : self.chem_dim + self.chemberta_dim]
        blocks = self._blocks(chem_x, cb_x)
        g = self._gates(torch.cat(blocks, dim=1))
        return {
            "block_names": list(self.block_names),
            "mean_per_block": [round(float(v), 4) for v in g.mean(dim=(0, 1))],
            "per_molecule_sd": round(float(g.std(dim=0).mean()), 4),
            "per_channel_spread": round(float(g.mean(dim=0).std(dim=0).mean()), 4),
        }

    def predict(self, x: Tensor) -> dict[str, Tensor]:
        """Head outputs in the units a caller actually wants.

        Regression heads come back as pChEMBL rather than z-scores; classification heads come
        back as probabilities. Use this rather than forward() anywhere the number leaves the
        training loop -- the downstream cardiotoxicity model consumes IC50 values, so a
        z-score silently substituted for a pChEMBL is the most expensive possible bug here.
        """
        raw = self.forward(x)
        out: dict[str, Tensor] = {}
        for i, name in enumerate(REGRESSION_HEADS):
            out[name] = raw[name] * self.reg_std[i] + self.reg_mean[i]
        for name in CLASSIFICATION_HEADS:
            out[name] = torch.sigmoid(raw[name])
        return out

    def tau_consistency(self, x: Tensor) -> dict[str, float]:
        """How often the free classification head agrees with thresholding its own regression
        head at the label's nominal boundary.

        Nothing in this architecture couples the two, so they can and do disagree. The number
        is a diagnostic, not a loss: forcing agreement by construction collapsed sensitivity
        to 7.2% in the internal tied arm, so disagreement here is a feature being measured
        rather than a bug being hunted.
        """
        pred = self.predict(x)
        reg_of = {
            "herg_blocker_10um": "herg_pchembl",
            "herg_blocker_1um": "herg_pchembl",
            "nav15_blocker": "nav15_pchembl",
            "cav12_blocker": "cav12_pchembl",
        }
        return {
            name: float(((pred[name] > 0.5) == (pred[reg_of[name]] > tau)).float().mean())
            for name, tau in NOMINAL_TAU.items()
        }

    def chem_embedding(self, x: Tensor) -> Tensor:
        chem_x = x[:, : self.chem_dim]
        return torch.cat(
            [self.fp_branch(chem_x[:, : self.fp_dim]), self.desc_branch(chem_x[:, self.fp_dim :])],
            dim=1,
        )

    def param_report(self) -> dict[str, int]:
        g = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {
            "fp_branch": g(self.fp_branch),
            "desc_branch": g(self.desc_branch),
            "chemberta_branch": g(self.chemberta_branch) if self.use_chemberta else 0,
            "gating": (g(self.desc_branches) + g(self.gate_net)) if self.attention == "gated" else 0,
            "attention": (
                g(self.key_proj) + g(self.value_proj) + self.query_embed.numel() + 1
                + (g(self.query_from_trunk) if self.attention in ("trunk", "centered") else 0)
                if self.attention in ("static", "trunk", "centered") else 0
            ),
            "heads": g(self.heads),
            "total": sum(p.numel() for p in self.parameters()),
        }


# Regression label column -> its index in REGRESSION_HEADS, matching the trainer's layout.
REG_LABEL_COL = {0: 0, 3: 1, 5: 2}

# State-dict key prefixes belonging to the cross-attention line, which this class cannot
# represent. Their presence means the file was written by a different model, so it is worth
# a better error than a shape clash. Any other foreign module surfaces as an unexpected key
# from load_state_dict, which names it plainly enough.
FOREIGN_MARKERS = ("chem_branch.", "attn.", "queries", "chem_proj.")


def load_checkpoint(model: CardioSafe, ck: dict) -> dict[str, list[str]]:
    """Load a checkpoint: weights AND the inference contract.

    A checkpoint is not just `model_state_dict`. It also carries reg_scalers -- the mean/std
    the regression targets were standardised by -- and the descriptor scaler. The weights
    alone will run and produce plausible-looking numbers in the wrong units, which is why
    this is the entry point.
    """
    if ck.get("swa_n"):
        raise ValueError(
            f"checkpoint has swa_n={ck['swa_n']} and an averaged swa_state_dict; decide which "
            "weights you mean and load that state dict directly"
        )
    sd = ck["model_state_dict"]
    stale = sorted({k for k in sd if k.startswith(FOREIGN_MARKERS)})
    if stale:
        raise ValueError(
            f"checkpoint carries {len(stale)} keys from the cross-attention architecture "
            f"(e.g. {stale[:3]}), which this class has no module for. Score it with the "
            "class that trained it."
        )
    model.load_state_dict(sd)
    report: dict[str, list[str]] = {"loaded": sorted(sd)}

    scalers = ck.get("reg_scalers") or {}
    for col, i in REG_LABEL_COL.items():
        if col in scalers:
            mean, std = scalers[col]
            model.reg_mean[i], model.reg_std[i] = float(mean), float(std)
    report["reg_scalers"] = sorted(str(c) for c in scalers)

    # The descriptor scaler is NOT loaded onto the model. Descriptors arrive already z-scored
    # from the data pipeline, so a copy on the module would be state that never gets read. It
    # is reported so a caller building features from raw SMILES knows it must apply these.
    dm = ck.get("descriptor_scaler_means")
    if dm is not None:
        report["desc_scaler_NOT_applied"] = [f"{len(dm)} columns, caller's responsibility"]

    return report
