"""A shared channel-conditioned decoder, with matched concat/static/attention arms.

Parameter sharing is a separate hypothesis from adaptive routing. Regression
and classification remain separate learned outputs; no hard threshold ties.
"""
import math
import torch
from torch import nn
from model.cardiosafe import _init_weights
from model.latent_fusion import CachedConcat


class ChannelAttention(nn.Module):
    def __init__(self, static=False, heads=4):
        super().__init__()
        if 64 % heads:
            raise ValueError('64 must be divisible by attention heads')
        self.static, self.heads = static, heads
        self.norm = nn.LayerNorm(64)
        self.query_trunk = nn.Sequential(nn.LayerNorm(448), nn.Linear(448, 64))
        self.query_channel = nn.Linear(16, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.output = nn.Linear(64, 448)
        self.log_temp = nn.Parameter(torch.full((heads,), math.log(2.0)))
        self.static_logits = nn.Parameter(torch.zeros(3, heads, 7))
        _init_weights(self)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if static:
            self.query_trunk.requires_grad_(False)
            self.query_channel.requires_grad_(False)
            self.key.requires_grad_(False)
            self.log_temp.requires_grad_(False)
        else:
            self.static_logits.requires_grad_(False)

    def routes(self, tokens, channel_codes):
        b = len(tokens)
        if self.static:
            return self.static_logits.softmax(-1)[None].expand(b, -1, -1, -1)
        q = self.query_trunk(tokens.flatten(1))[:, None] + self.query_channel(channel_codes)[None]
        q = nn.functional.normalize(q.reshape(b, 3, self.heads, -1), dim=-1)
        k = nn.functional.normalize(self.key(self.norm(tokens)).reshape(b, 7, self.heads, -1), dim=-1)
        scores = torch.einsum('bchd,bthd->bcht', q, k)
        return (scores*self.log_temp.clamp(-2, 3).exp()[None, None, :, None]).softmax(-1)

    def forward(self, tokens, weights):
        v = self.value(self.norm(tokens)).reshape(len(tokens), 7, self.heads, -1)
        context = torch.einsum('bcht,bthd->bchd', weights, v).flatten(2)
        return self.output(context)


class SharedFusion(CachedConcat):
    MODES = ('shared_concat', 'shared_attention', 'shared_static')

    def __init__(self, mode='shared_attention', heads=4, dropout=.3):
        super().__init__(attention='none', dropout=dropout)
        if mode not in self.MODES:
            raise ValueError(mode)
        self.attention = self.mode = mode
        self.heads = nn.ModuleDict()  # remove the original seven independent heads
        self.token_names = [f'fingerprint_latent_{i}' for i in range(4)] + [
            'descriptors', 'chemberta_latent_0', 'chemberta_latent_1']
        with torch.random.fork_rng(devices=[]):
            self.channel_codes = nn.Embedding(3, 16)
            self.decoder = nn.Sequential(nn.Linear(464, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 3))
            _init_weights(self.decoder)
            self.fusion = None if mode == 'shared_concat' else ChannelAttention(static=mode == 'shared_static', heads=heads)

    def tokens(self, x):
        return torch.cat([self.fp_branch(x[:, :6144]), self.desc_branch(x[:, 6144:6164]),
                          self.chemberta_branch(x[:, 6164:6548])], 1).reshape(-1, 7, 64)

    def routes(self, tokens):
        return self.fusion.routes(tokens, self.channel_codes.weight)

    def from_tokens(self, tokens, routing=None):
        b = len(tokens)
        trunk = tokens.flatten(1)[:, None].expand(-1, 3, -1)
        if self.fusion is not None:
            trunk = trunk + self.fusion(tokens, self.routes(tokens) if routing is None else routing)
        codes = self.channel_codes.weight[None].expand(b, -1, -1)
        out = self.decoder(torch.cat([trunk, codes], -1))
        return {'herg_pchembl':out[:,0,0], 'nav15_pchembl':out[:,1,0], 'cav12_pchembl':out[:,2,0],
                'herg_blocker_10um':out[:,0,1], 'herg_blocker_1um':out[:,0,2],
                'nav15_blocker':out[:,1,1], 'cav12_blocker':out[:,2,1]}

    def forward_with_embedding(self, x):
        branches, embedding = self._cached_branches(x)
        return self.from_tokens(torch.cat(branches, 1).reshape(-1, 7, 64)), embedding

    @torch.no_grad()
    def attention_stats(self, x):
        if self.fusion is None:
            return {}
        w = self.routes(self.tokens(x))
        return {'per_molecule_sd':float(w.std(0).mean()),
                'entropy':float(-(w*w.clamp_min(1e-9).log()).sum(-1).mean()),
                'mean_weights':w.mean((0,1,2)).tolist(), 'token_names':self.token_names}

    def param_report(self):
        r = super().param_report()
        r['heads'] = sum(p.numel() for p in self.decoder.parameters()) + self.channel_codes.weight.numel()
        r['attention'] = 0 if self.fusion is None else sum(p.numel() for p in self.fusion.parameters())
        r['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return r
