"""Matched attention comparator for the author's locked concat architecture."""
import torch
from torch import nn
from model.cardiosafe import CardioSafe,_init_weights
from model.revision import build_revision_model


class RevisionAttention(CardioSafe):
    channels=('herg','nav15','cav12','iks')

    def __init__(self,mode='static',chem_dim=6164,encoder_dim=583,chemberta=True,dropout=.3):
        super().__init__(chem_dim=chem_dim,chemberta_dim=encoder_dim,chemberta=chemberta,
                         dropout=dropout,attention=mode)
        self.query_embed=nn.Parameter(torch.cat([self.query_embed.detach(),torch.randn(1,self.query_embed.shape[1])*.02]))
        if mode=='centered':self.logit_norm=nn.BatchNorm1d(4*len(self.key_proj),affine=False)
        width=self.heads['herg_blocker_10um'][0].in_features
        self.heads['iks_blocker']=nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,1))
        _init_weights(self.heads['iks_blocker'])

    def forward_with_embedding(self,x):
        chemical=x[:,:self.chem_dim];encoder=x[:,self.chem_dim:self.chem_dim+self.chemberta_dim]
        branches=[self.fp_branch(chemical[:,:self.fp_dim]),self.desc_branch(chemical[:,self.fp_dim:])]
        embedding=torch.cat(branches,1)
        if self.use_chemberta:branches.append(self.chemberta_branch(encoder))
        shared=torch.cat(branches,1);context,_=self._attend(branches,shared)
        return {name:head(torch.cat([shared,context[:,self.channels.index(name.split('_')[0])]],1)).squeeze(-1)
                for name,head in self.heads.items()},embedding


def build_baseline_model(mode='none',slots=4,chemberta=True,chem_dim=6164,dropout=.3,packed_tokens=False,encoder_dim=583):
    if mode=='none':return build_revision_model(mode,slots,chemberta,chem_dim,dropout,packed_tokens,encoder_dim)
    if packed_tokens or mode not in ('static','trunk','centered'):raise ValueError('Unsupported revision comparator')
    return RevisionAttention(mode,chem_dim,encoder_dim,chemberta,dropout)
