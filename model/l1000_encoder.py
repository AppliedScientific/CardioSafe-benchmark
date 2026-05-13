"""L1000 gene-expression encoder.

GCN + per-gene FFN architecture that predicts 978 LINCS L1000 landmark-gene
expression z-scores from a 1024-dim count TopologicalTorsion fingerprint, as
described in the Methods section of Jovanovic et al. 2026.

Topology:
    gene_embedding: nn.Embedding(978, 384)
    2 x GCNConv(384, 384) over the gene co-expression graph
        (edges where |Pearson r| > 0.40 on training-set L1000 signatures;
        graph is passed in by the caller as `edge_index`)
    Shared FFN over [gene_emb || mol_emb]:
        Linear(384 + 1024, 512) -> ReLU -> Dropout(0.2)
        Linear(512, 512) -> ReLU -> Dropout(0.2)
        Linear(512, 512) -> ReLU -> Dropout(0.2)
    Per-gene linear head: (978 x 512) weight + (978,) bias, output (B, 978)

The trained encoder is run once per compound and the resulting 978-dim
prediction is consumed as the L1000 branch input to CardioSafe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class L1000ExpressionEncoder(nn.Module):
    """GCN + per-gene FFN on a 1024-dim molecular fingerprint."""

    def __init__(
        self,
        n_genes: int = 978,
        mol_dim: int = 1024,
        gene_dim: int = 384,
        dropout: float = 0.2,
        *,
        edge_index: Tensor,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.mol_dim = mol_dim
        self.gene_dim = gene_dim

        self.register_buffer("gene_ids", torch.arange(n_genes, dtype=torch.long))
        self.gene_embedding = nn.Embedding(n_genes, gene_dim)
        self.gcn1 = GCNConv(gene_dim, gene_dim)
        self.gcn2 = GCNConv(gene_dim, gene_dim)
        self.register_buffer("edge_index", edge_index)

        ffn_input = mol_dim + gene_dim
        self.ffn1 = nn.Linear(ffn_input, 512)
        self.ffn2 = nn.Linear(512, 512)
        self.ffn3 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout)

        self.gene_out_weight = nn.Parameter(torch.empty(n_genes, 512))
        self.gene_out_bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, mol_emb: Tensor) -> Tensor:
        batch_size = mol_emb.shape[0]

        gene_emb = self.gene_embedding(self.gene_ids)
        gene_emb = self.gcn1(gene_emb, self.edge_index)
        gene_emb = self.gcn2(gene_emb, self.edge_index)

        gene_exp = gene_emb.unsqueeze(0).expand(batch_size, self.n_genes, self.gene_dim)
        mol_exp = mol_emb.unsqueeze(1).expand(batch_size, self.n_genes, self.mol_dim)
        x = torch.cat([gene_exp, mol_exp], dim=2)

        x = F.relu(self.ffn1(x))
        x = self.dropout(x)
        x = F.relu(self.ffn2(x))
        x = self.dropout(x)
        x = F.relu(self.ffn3(x))
        x = self.dropout(x)

        return (x * self.gene_out_weight.unsqueeze(0)).sum(dim=-1) + self.gene_out_bias
