"""Experimental compute optimization; not used by the running Phase-5 models.

Project only valid token rows, then restore the small key/value tensors for
masked attention. Parameter names and mathematical readout match the padded pool.
Numerical/gradient equivalence must be verified before using it with checkpoints.
"""
import math
import torch
from model.molecular_token_fusion import MolecularTokenPool


class PackedMolecularTokenPool(MolecularTokenPool):
    @staticmethod
    def valid_states(tokens, mask):
        if tokens.ndim != 3 or tokens.shape[-1] != 384 or mask.shape != tokens.shape[:2]:
            raise ValueError('Expected tokens [batch,length,384] and matching mask')
        if mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError('Every molecule needs a nonempty boolean token mask')
        return tokens[mask]

    def packed_routes(self, chemical_context, z, mask, queries):
        batch,length=mask.shape
        if chemical_context.shape!=(batch,320):
            raise ValueError('Chemical context and token batches must align')
        if self.mode=='mean':
            scores=z.new_zeros((batch,3,length))
        else:
            q=self.queries(chemical_context) if queries is None else queries
            if q.shape!=(batch,3,self.query_width):raise ValueError('Query intervention has the wrong shape')
            keys=z.new_zeros((batch,length,self.query_width))
            keys[mask]=self.key(z)
            scores=torch.einsum('bck,blk->bcl',q,keys)/math.sqrt(self.query_width)
        return scores.masked_fill(~mask[:,None,:],float('-inf')).softmax(-1)

    def routes(self, chemical_context, tokens, mask, queries=None):
        z=self.token_norm(self.valid_states(tokens,mask))
        return self.packed_routes(chemical_context,z,mask,queries)

    def forward(self, chemical_context, tokens, mask, queries=None, routing=None):
        z=self.token_norm(self.valid_states(tokens,mask))
        weights=self.packed_routes(chemical_context,z,mask,queries) if routing is None else routing
        if weights.shape!=(len(tokens),3,tokens.shape[1]):raise ValueError('Route intervention has the wrong shape')
        if (not torch.isfinite(weights).all() or (weights<0).any()
                or weights.masked_select(~mask[:,None,:].expand_as(weights)).count_nonzero()
                or not torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=1e-5,rtol=0)):
            raise ValueError('Routes must be masked probability distributions')
        values=z.new_zeros((*mask.shape,self.value_width))
        values[mask]=self.value(z)
        context=torch.einsum('bcl,blv->bcv',weights,values)
        return self.output(context),weights
