"""Context-conditioned attention over active fingerprint bits before encoding.

The 6144 fixed columns represent three variable-size sets of active hash bits.
We mask absent bits and normalize within each fingerprint family. Multiplying
attention probabilities by the active count preserves the baseline sum and
cardinality at initialization. Hash bits are not unique chemical substructures.
Dense masked computation avoids costly sparse embedding-bag CPU backward.
"""
import math
import torch
from torch import nn
from model.cardiosafe import _init_weights
from model.latent_fusion import CachedConcat


class BitRouter(nn.Module):
    def __init__(self, static=False, key_dim=16):
        super().__init__()
        self.static, self.key_dim = static, key_dim
        self.query = nn.Sequential(nn.Linear(404, 32), nn.ReLU(), nn.Linear(32, 3*key_dim))
        self.keys = nn.Parameter(torch.randn(3, 2048, key_dim))
        self.bias = nn.Parameter(torch.zeros(3, 2048))
        _init_weights(self)
        # Must follow generic initialization. Uniform active-token attention
        # means multiplier EXACTLY 1, not a small perturbation of concat.
        nn.init.zeros_(self.query[-1].weight)
        nn.init.zeros_(self.query[-1].bias)
        if static:
            self.query.requires_grad_(False)
            self.keys.requires_grad_(False)

    def queries(self, x):
        if self.static:
            return x.new_zeros(len(x), 3, self.key_dim)
        context = torch.cat([x[:, 6144:6164], nn.functional.layer_norm(x[:, 6164:6548], (384,))], 1)
        return self.query(context).reshape(-1, 3, self.key_dim)

    def gates(self, x, queries=None, uniform=False):
        active = x[:, :6144].reshape(-1, 3, 2048).ne(0)
        if uniform:
            return active.to(x.dtype).flatten(1)
        q = self.queries(x) if queries is None else queries
        scores = torch.einsum('bfd,ftd->bft', q, self.keys) / math.sqrt(self.key_dim) + self.bias
        scores = scores.masked_fill(~active, -torch.inf)
        maximum = scores.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        numer = (scores-maximum).exp()
        count = active.sum(-1, keepdim=True)
        gates = numer * count / numer.sum(-1, keepdim=True).clamp_min(1e-30)
        return gates.flatten(1)


class BitFusion(CachedConcat):
    MODES = ('bit_attention', 'bit_static')

    def __init__(self, mode='bit_attention', dropout=.3):
        super().__init__(attention='none', dropout=dropout)
        if mode not in self.MODES:
            raise ValueError(mode)
        self.attention = self.mode = mode
        with torch.random.fork_rng(devices=[]):
            self.router = BitRouter(static=mode == 'bit_static')

    def queries(self, x):
        return self.router.queries(x)

    def forward_with_queries(self, x, queries=None, uniform=False, return_embedding=False):
        gates = self.router.gates(x, queries, uniform)
        hidden = self.fp_branch[0](x[:, :6144] * gates)
        fp_rest = self.fp_branch[1:]
        embedding = torch.cat([fp_rest(hidden), self.desc_branch(x[:, 6144:6164])], 1)
        shared = torch.cat([fp_rest(hidden), self.desc_branch(x[:, 6144:6164]),
                            self.chemberta_branch(x[:, 6164:6548])], 1)
        out = {h: head(shared).squeeze(-1) for h, head in self.heads.items()}
        return (out, embedding) if return_embedding else out

    def forward_with_embedding(self, x):
        return self.forward_with_queries(x, return_embedding=True)

    @torch.no_grad()
    def attention_stats(self, x):
        # Conditional on presence: absent bits/variable set sizes must not create
        # a fake routing SD at initialization. Chunking bounds diagnostic memory.
        count = x[:, :6144].ne(0).sum(0)
        total = torch.zeros(6144, dtype=torch.float64, device=x.device)
        squares = torch.zeros_like(total)
        for chunk in x.split(256):
            g = self.router.gates(chunk).double()
            total += g.sum(0)
            squares += g.square().sum(0)
        mean = total / count.clamp_min(1)
        var = ((squares - total.square()/count.clamp_min(1))/(count-1).clamp_min(1)).clamp_min(0)
        valid = count > 1
        return {'per_molecule_sd': float(var[valid].sqrt().mean()),
                'sd_definition': 'mean bit multiplier SD conditional on bit presence, bits with >=2 observations',
                'gate_rms_from_one': float(((squares.sum()-2*total.sum()+count.sum())/count.sum().clamp_min(1)).clamp_min(0).sqrt()),
                'observed_bits': int((count > 0).sum()),
                'mean_multiplier_by_family': [float(total[i:i+2048].sum()/count[i:i+2048].sum().clamp_min(1)) for i in (0, 2048, 4096)]}

    def param_report(self):
        report = super().param_report()
        report['attention'] = sum(p.numel() for p in self.router.parameters())
        report['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report
