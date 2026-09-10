"""Attention over the original 448-wide concat trunk, without changing encoders.

Four FP latent chunks, one descriptor chunk, two ChemBERTa chunks. These are
learned coordinates, not claims about identifiable chemical substructures.
"""
import math
import torch
from torch import nn
from model.cardiosafe import CardioSafe, _init_weights


class CachedConcat(CardioSafe):
    """Same original function, BN updates and dropout order; reuse the first FP Linear.

    The historical forward evaluates that deterministic Linear twice on exactly
    the same input. All downstream stochastic layers still execute twice.
    """
    def _cached_branches(self, x):
        chem, cb = x[:, :self.chem_dim], x[:, self.chem_dim:self.chem_dim+self.chemberta_dim]
        fp_hidden = self.fp_branch[0](chem[:, :self.fp_dim])
        fp_rest = self.fp_branch[1:]
        embedding = torch.cat([fp_rest(fp_hidden), self.desc_branch(chem[:, self.fp_dim:])], 1)
        branches = [fp_rest(fp_hidden), self.desc_branch(chem[:, self.fp_dim:])]
        if self.use_chemberta:
            branches.append(self.chemberta_branch(cb))
        return branches, embedding

    def forward_with_embedding(self, x):
        branches, embedding = self._cached_branches(x)
        shared = torch.cat(branches, 1)
        return {h: head(shared).squeeze(-1) for h, head in self.heads.items()}, embedding


class LatentAttention(nn.Module):
    def __init__(self, static=False, width=64, heads=4, tokens=7):
        super().__init__()
        self.static, self.width, self.heads = static, width, heads
        if width % heads:
            raise ValueError('Width must be divisible by heads')
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.log_temp = nn.Parameter(torch.full((heads,), math.log(2.0)))
        self.static_logits = nn.Parameter(torch.zeros(heads, tokens, tokens))
        _init_weights(self)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if static:
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)
            self.log_temp.requires_grad_(False)
        else:
            self.static_logits.requires_grad_(False)

    def routes(self, tokens):
        b, n, _ = tokens.shape
        if self.static:
            return self.static_logits.softmax(-1)[None].expand(b, -1, -1, -1)
        z = self.norm(tokens)
        q = self.query(z).reshape(b, n, self.heads, -1).transpose(1, 2)
        k = self.key(z).reshape(b, n, self.heads, -1).transpose(1, 2)
        scores = nn.functional.normalize(q, dim=-1) @ nn.functional.normalize(k, dim=-1).transpose(-1, -2)
        return (scores*self.log_temp.clamp(-2, 3).exp()[None, :, None, None]).softmax(-1)

    def forward(self, tokens, routing=None):
        b, n, _ = tokens.shape
        weights = self.routes(tokens) if routing is None else routing
        values = self.value(self.norm(tokens)).reshape(b, n, self.heads, -1).transpose(1, 2)
        context = (weights @ values).transpose(1, 2).reshape(b, n, self.width)
        return tokens + self.output(context)


class LatentFusion(CachedConcat):
    MODES = ('latent_self', 'latent_static')

    def __init__(self, mode='latent_self', heads=4, dropout=.3):
        super().__init__(attention='none', dropout=dropout)
        if mode not in self.MODES:
            raise ValueError(mode)
        self.attention = self.mode = mode
        self.token_names = [f'fingerprint_latent_{i}' for i in range(4)] + [
            'descriptors', 'chemberta_latent_0', 'chemberta_latent_1']
        # Preserve the baseline RNG state too. There is no extra dropout on the
        # residual, so even the first training forward matches concat exactly.
        with torch.random.fork_rng(devices=[]):
            self.fusion = LatentAttention(static=mode == 'latent_static', heads=heads)

    def tokens(self, x):
        chem, cb = x[:, :self.chem_dim], x[:, self.chem_dim:self.chem_dim+self.chemberta_dim]
        return torch.cat([self.fp_branch(chem[:, :self.fp_dim]),
                          self.desc_branch(chem[:, self.fp_dim:]), self.chemberta_branch(cb)], 1).reshape(-1, 7, 64)

    def routes(self, tokens):
        return self.fusion.routes(tokens)

    def from_tokens(self, tokens, routing=None):
        shared = self.fusion(tokens, routing).flatten(1)
        return {h: head(shared).squeeze(-1) for h, head in self.heads.items()}

    def forward_with_embedding(self, x):
        branches, embedding = self._cached_branches(x)
        tokens = torch.cat(branches, 1).reshape(-1, 7, 64)
        return self.from_tokens(tokens), embedding

    @torch.no_grad()
    def attention_stats(self, x):
        w = self.routes(self.tokens(x))
        return {'per_molecule_sd': float(w.std(0).mean()),
                'entropy': float(-(w*w.clamp_min(1e-9).log()).sum(-1).mean()),
                'mean_weights': w.mean((0, 1, 2)).tolist(), 'token_names': self.token_names}

    def param_report(self):
        report = super().param_report()
        report['attention'] = sum(p.numel() for p in self.fusion.parameters())
        report['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report
