"""Directed-bond MPNN comparator, implemented here rather than Chemprop CLI.

Uses the Chemprop D-MPNN update (reverse-edge exclusion), depth3, hidden300,
mean atom pooling and the same eight endpoint heads. No pretrained weights.
Reference: https://chemprop.readthedocs.io/en/latest/tutorial/python/models/message_passing.html
"""
import torch
from torch import nn
from model.revision import REVISION_HEADS


class RevisionDMPNN(nn.Module):
    def __init__(self,hidden=300,depth=3,dropout=.3):
        super().__init__();self.depth=depth;self.hidden=hidden
        self.edge_input=nn.Linear(133+14,hidden,bias=False)
        self.edge_update=nn.Linear(hidden,hidden,bias=False)
        self.atom_output=nn.Linear(133+hidden,hidden)
        self.dropout=nn.Dropout(dropout)
        self.heads=nn.ModuleDict({h:nn.Sequential(nn.Linear(hidden,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,1)) for h in REVISION_HEADS})
        self.register_buffer('reg_mean',torch.zeros(3));self.register_buffer('reg_std',torch.ones(3))

    def encode(self,graph):
        atoms,bonds,src,dst,reverse,batch,n=graph
        initial=torch.relu(self.edge_input(torch.cat([atoms[src],bonds],1)));h=initial
        for _ in range(self.depth-1):
            incoming=h.new_zeros((len(atoms),self.hidden)).index_add(0,dst,h)
            message=incoming[src]-h[reverse]
            h=self.dropout(torch.relu(initial+self.edge_update(message)))
        incoming=h.new_zeros((len(atoms),self.hidden)).index_add(0,dst,h)
        nodes=self.dropout(torch.relu(self.atom_output(torch.cat([atoms,incoming],1))))
        sums=nodes.new_zeros((n,self.hidden)).index_add(0,batch,nodes)
        counts=nodes.new_zeros(n).index_add(0,batch,torch.ones(len(atoms),device=nodes.device))
        return sums/counts[:,None].clamp_min(1)

    def forward_graph(self,graph):
        h=self.encode(graph);return {name:head(h).squeeze(-1) for name,head in self.heads.items()}
