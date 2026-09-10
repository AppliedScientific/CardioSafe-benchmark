"""Controlled, small token-fusion models for the phase-4 investigation.

Eight semantically defined tokens: three fingerprint families, four descriptor
groups, and ChemBERTa. Unequal input widths have separate encoders into width 64.
Slots are independent queries, not duplicated copies of a pooled context. All
token arms use identical encoders. No test/batch statistics normalize routing.
"""
from __future__ import annotations

import math
import torch
from torch import nn, Tensor

from model.cardiosafe import (ALL_HEADS, CHANNELS, HEAD_CHANNEL, DESCRIPTOR_BLOCKS,
                             REGRESSION_HEADS, CLASSIFICATION_HEADS, _init_weights)


class TokenFusion(nn.Module):
    MODES = ("token_concat", "slots", "slot_static", "slot_residual", "token_mlp",
             "token_self", "token_self_static", "balanced_slots")

    def __init__(self, mode="slots", width=64, slots=4, key_dim=16, dropout=0.3):
        super().__init__()
        encoder_init_state = torch.get_rng_state()
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode, self.width, self.slots, self.key_dim = mode, width, slots, key_dim
        self.token_names = ["morgan", "atompair", "toptorsion", *DESCRIPTOR_BLOCKS, "chemberta"]
        indices = [list(range(i * 2048, (i + 1) * 2048)) for i in range(3)]
        indices += [[6144 + i for i in cols] for cols in DESCRIPTOR_BLOCKS.values()]
        indices += [list(range(6164, 6548))]
        self.encoders = nn.ModuleList()
        for i, cols in enumerate(indices):
            self.register_buffer(f"cols_{i}", torch.tensor(cols), persistent=False)
            hidden = 128 if i < 3 else 64
            self.encoders.append(nn.Sequential(
                nn.Linear(len(cols), hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Dropout(0.2 if i < 3 or i == 7 else 0.1),
                nn.Linear(hidden, width), nn.BatchNorm1d(width), nn.ReLU(),
            ))
        n = len(indices)
        self.shared_dim = n * width
        if mode in ("token_self", "token_self_static"):
            if width % slots:
                raise ValueError('Token width must be divisible by attention heads')
            self.key = nn.Linear(width, width)
            self.query = nn.Linear(width, width)
            self.value = nn.Linear(width, width)
            self.token_norm = nn.LayerNorm(width)
            self.token_out = nn.Linear(width, width)
            self.token_dropout = nn.Dropout(0.1)
            self.log_temp = nn.Parameter(torch.full((slots,), math.log(2.0)))
            self.static_logits = nn.Parameter(torch.zeros(slots, n, n))
            if mode == "token_self_static":
                self.key.requires_grad_(False)
                self.query.requires_grad_(False)
                self.log_temp.requires_grad_(False)
            else:
                self.static_logits.requires_grad_(False)
        if mode in ("slots", "slot_static", "slot_residual", "balanced_slots"):
            self.key = nn.Linear(width, slots * key_dim)
            self.query = nn.Linear(self.shared_dim, slots * key_dim)
            self.query_embed = nn.Parameter(torch.randn(3, slots, key_dim) * 0.2)
            self.value = nn.Linear(width, width)
            self.log_temp = nn.Parameter(torch.full((slots,), math.log(2.0)))
            self.static_logits = nn.Parameter(torch.zeros(3, slots, n))
            if mode == "slot_static":
                for module in (self.key, self.query):
                    module.requires_grad_(False)
                self.query_embed.requires_grad_(False)
                self.log_temp.requires_grad_(False)
            else:
                self.static_logits.requires_grad_(False)
        if mode == "token_mlp":
            self.extra = nn.Sequential(nn.Linear(self.shared_dim, slots * width), nn.ReLU())
        head_dim = self.shared_dim if mode in ("token_concat", "token_self", "token_self_static") else slots * width
        if mode in ("slot_residual", "token_mlp"):
            head_dim += self.shared_dim
        self.heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(head_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        ) for name in ALL_HEADS})
        self.register_buffer("reg_mean", torch.zeros(3))
        self.register_buffer("reg_std", torch.ones(3))
        _init_weights(self)
        # Match encoder initial weights across arms despite different fusion/head
        # construction consuming different numbers of random values.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(encoder_init_state)
            _init_weights(self.encoders)
        if mode in ("token_self", "token_self_static"):
            # Residual starts at zero, AFTER generic Linear initialization.
            nn.init.zeros_(self.token_out.weight)
            nn.init.zeros_(self.token_out.bias)
            # Exactly match the established token-concat heads as well as its
            # encoders. Creating the reference inside fork_rng preserves the RNG.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(encoder_init_state)
                reference = TokenFusion(mode="token_concat", width=width, slots=slots,
                                        key_dim=key_dim, dropout=dropout)
            self.heads.load_state_dict(reference.heads.state_dict())

    def tokens(self, x: Tensor) -> Tensor:
        return torch.stack([enc(x[:, getattr(self, f"cols_{i}")])
                            for i, enc in enumerate(self.encoders)], dim=1)

    def routes(self, tokens: Tensor) -> Tensor:
        b, n, _ = tokens.shape
        if self.mode in ("token_self", "token_self_static"):
            if self.mode == "token_self_static":
                return self.static_logits.softmax(-1).unsqueeze(0).expand(b, -1, -1, -1)
            z = self.token_norm(tokens)
            q = self.query(z).reshape(b, n, self.slots, -1).transpose(1, 2)
            k = self.key(z).reshape(b, n, self.slots, -1).transpose(1, 2)
            logits = nn.functional.normalize(q, dim=-1) @ nn.functional.normalize(k, dim=-1).transpose(-1, -2)
            return (logits * self.log_temp.clamp(-2, 3).exp()[None, :, None, None]).softmax(-1)
        if self.mode == "slot_static":
            return self.static_logits.softmax(-1).unsqueeze(0).expand(b, -1, -1, -1)
        key = self.key(tokens).reshape(b, n, self.slots, self.key_dim)
        query = self.query(tokens.flatten(1)).reshape(b, 1, self.slots, self.key_dim)
        channel = self.query_embed.unsqueeze(0)
        if self.mode == 'balanced_slots':
            query = nn.functional.normalize(query, dim=-1)
            channel = nn.functional.normalize(channel, dim=-1)
        query = query + channel
        key = nn.functional.normalize(key, dim=-1)
        query = nn.functional.normalize(query, dim=-1)
        logits = torch.einsum("bcsd,bnsd->bcsn", query, key)
        return (logits * self.log_temp.clamp(-2, 3).exp()[None, None, :, None]).softmax(-1)

    def from_tokens(self, tokens: Tensor, routing: Tensor | None = None):
        shared = tokens.flatten(1)
        if self.mode in ("token_self", "token_self_static"):
            b, n, _ = tokens.shape
            weights = self.routes(tokens) if routing is None else routing
            values = self.value(self.token_norm(tokens)).reshape(b, n, self.slots, -1).transpose(1, 2)
            context = (weights @ values).transpose(1, 2).reshape(b, n, self.width)
            updated = tokens + self.token_dropout(self.token_out(context))
            return {h: module(updated.flatten(1)).squeeze(-1) for h, module in self.heads.items()}
        if self.mode in ("token_concat", "token_mlp"):
            if self.mode == "token_mlp":
                shared = torch.cat([shared, self.extra(shared)], -1)
            return {h: module(shared).squeeze(-1) for h, module in self.heads.items()}
        weights = self.routes(tokens) if routing is None else routing
        context = torch.einsum("bcsn,bnd->bcsd", weights, self.value(tokens)).flatten(2)
        return {h: module(torch.cat([shared, context[:, CHANNELS.index(HEAD_CHANNEL[h])]], -1)
                          if self.mode == "slot_residual" else
                          context[:, CHANNELS.index(HEAD_CHANNEL[h])]).squeeze(-1)
                for h, module in self.heads.items()}

    def forward(self, x: Tensor):
        return self.from_tokens(self.tokens(x))

    def predict(self, x: Tensor):
        """pChEMBL regression and classification probabilities, for scaled features."""
        raw = self(x)
        return {**{h: raw[h] * self.reg_std[i] + self.reg_mean[i]
                   for i, h in enumerate(REGRESSION_HEADS)},
                **{h: raw[h].sigmoid() for h in CLASSIFICATION_HEADS}}

    @torch.no_grad()
    def attention_stats(self, x: Tensor):
        if self.mode in ("token_concat", "token_mlp"):
            return {}
        w = self.routes(self.tokens(x))
        return {"per_molecule_sd": w.std(0).mean().item(),
                "entropy": -(w * w.clamp_min(1e-9).log()).sum(-1).mean().item(),
                "mean_weights": w.mean((0, 1, 2)).tolist(), "token_names": self.token_names,
                "slot_sd": w.std(2).mean().item() if self.slots > 1 else 0.0}

    def gate_stats(self, x):
        return {}

    def param_report(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "attention": sum(p.numel() for name, p in self.named_parameters()
                                 if name.startswith(("query", "key", "value", "log_temp", "static_logits", "token_norm", "token_out")))}
