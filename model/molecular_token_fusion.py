"""Experimental attention over frozen molecular tokens before pooling.

The explicit token training interface keeps fixed-width features separate.
The feature matrix supplies its chosen 384-wide readout; the token input
adds local information. All modes preserve the original concat function at
initialization through a zero-initialized residual output projection.
"""
import math
import torch
from torch import nn
from model.cardiosafe import CHANNELS, HEAD_CHANNEL, REGRESSION_HEADS, CLASSIFICATION_HEADS, _init_weights
from model.latent_fusion import CachedConcat


class MolecularTokenPool(nn.Module):
    MODES = ('conditioned', 'fixed_query', 'mean')
    # Approximately match trainable adapter budgets without tuning on metrics.
    DEFAULT_VALUE_WIDTH = {'conditioned':32, 'fixed_query':63, 'mean':75}

    def __init__(self, mode='conditioned', query_width=16, value_width=None):
        super().__init__()
        if mode not in self.MODES: raise ValueError(mode)
        if query_width < 1: raise ValueError('Query width must be positive')
        value_width = self.DEFAULT_VALUE_WIDTH[mode] if value_width is None else value_width
        if value_width < 1: raise ValueError('Value width must be positive')
        self.mode, self.query_width, self.value_width = mode, query_width, value_width
        self.token_norm = nn.LayerNorm(384)
        self.context_norm = nn.LayerNorm(320)
        self.key = nn.Linear(384, query_width, bias=False)
        self.query = nn.Linear(320, len(CHANNELS)*query_width)
        self.value = nn.Sequential(nn.Linear(384,value_width),nn.GELU())
        self.output = nn.Linear(value_width,128,bias=False)
        _init_weights(self)
        nn.init.zeros_(self.query.weight)
        nn.init.normal_(self.query.bias,std=.02)
        nn.init.zeros_(self.output.weight)  # Last: generic initializer must not overwrite it.
        if mode != 'conditioned':
            self.query.weight.requires_grad_(False)
            self.context_norm.requires_grad_(False)
        if mode == 'mean':
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)

    @staticmethod
    def prepare(tokens, mask):
        if tokens.ndim != 3 or tokens.shape[-1] != 384 or mask.shape != tokens.shape[:2]:
            raise ValueError('Expected tokens [batch,length,384] and matching mask')
        if mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError('Every molecule needs a nonempty boolean token mask')
        return tokens.masked_fill(~mask[...,None],0)

    def queries(self, chemical_context):
        if chemical_context.ndim != 2 or chemical_context.shape[1] != 320:
            raise ValueError('Expected 320-wide chemical branch context')
        if self.mode == 'conditioned':
            q = self.query(self.context_norm(chemical_context))
        else:
            q = self.query.bias[None].expand(len(chemical_context),-1)
        return q.reshape(-1,len(CHANNELS),self.query_width)

    def routes(self, chemical_context, tokens, mask, queries=None):
        z = self.token_norm(self.prepare(tokens,mask))
        return self._routes(chemical_context,z,mask,queries)

    def _routes(self, chemical_context, z, mask, queries):
        if chemical_context.shape != (len(z),320):
            raise ValueError('Chemical context and token batches must align')
        if self.mode == 'mean':
            scores = z.new_zeros((len(z),len(CHANNELS),z.shape[1]))
        else:
            q = self.queries(chemical_context) if queries is None else queries
            if q.shape != (len(z),len(CHANNELS),self.query_width):
                raise ValueError('Query intervention has the wrong shape')
            scores = torch.einsum('bck,blk->bcl',q,self.key(z))/math.sqrt(self.query_width)
        return scores.masked_fill(~mask[:,None,:],float('-inf')).softmax(-1)

    def forward(self, chemical_context, tokens, mask, queries=None, routing=None):
        clean = self.prepare(tokens,mask)
        z = self.token_norm(clean)
        weights = self._routes(chemical_context,z,mask,queries) if routing is None else routing
        if weights.shape != (len(tokens),len(CHANNELS),tokens.shape[1]):
            raise ValueError('Route intervention has the wrong shape')
        if (not torch.isfinite(weights).all() or (weights < 0).any()
                or weights.masked_select(~mask[:,None,:].expand_as(weights)).count_nonzero()
                or not torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=1e-5,rtol=0)):
            raise ValueError('Routes must be masked probability distributions')
        values = self.value(z)
        context = torch.einsum('bcl,blv->bcv',weights,values)
        return self.output(context), weights


class MolecularTokenFusion(CachedConcat):
    """Original encoders/heads with a channel-specific token residual in CB space."""
    def __init__(self, mode='conditioned', query_width=16, value_width=None, dropout=.3, packed_tokens=False):
        super().__init__(attention='none',dropout=dropout)
        self.token_mode = mode
        self.packed_tokens = packed_tokens
        pool_class = MolecularTokenPool
        if packed_tokens:
            from model.packed_molecular_tokens import PackedMolecularTokenPool
            pool_class = PackedMolecularTokenPool
        with torch.random.fork_rng(devices=[]):
            self.token_pool = pool_class(mode,query_width,value_width)

    def forward_with_embedding(self, x, tokens, mask, queries=None, routing=None):
        branches, embedding = self._cached_branches(x)
        chemical = torch.cat(branches[:2],dim=1)
        delta, _ = self.token_pool(chemical,tokens,mask,queries,routing)
        trunks = [torch.cat([chemical,branches[2]+delta[:,c]],dim=1) for c in range(len(CHANNELS))]
        output = {name:head(trunks[CHANNELS.index(HEAD_CHANNEL[name])]).squeeze(-1)
                  for name,head in self.heads.items()}
        return output, embedding

    def forward(self, x, tokens, mask, queries=None, routing=None):
        return self.forward_with_embedding(x,tokens,mask,queries,routing)[0]

    def queries(self, x):
        branches, _ = self._cached_branches(x)
        return self.token_pool.queries(torch.cat(branches[:2],dim=1))

    def predict(self, x, tokens, mask, **kwargs):
        raw = self.forward(x,tokens,mask,**kwargs)
        return {**{h:raw[h]*self.reg_std[i]+self.reg_mean[i] for i,h in enumerate(REGRESSION_HEADS)},
                **{h:raw[h].sigmoid() for h in CLASSIFICATION_HEADS}}

    def param_report(self):
        report = super().param_report()
        report['attention'] = sum(p.numel() for p in self.token_pool.parameters())
        report['token_adapter_trainable'] = sum(p.numel() for p in self.token_pool.parameters() if p.requires_grad)
        report['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report
