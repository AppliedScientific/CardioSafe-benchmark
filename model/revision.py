"""Paper revision: chemical branches + final CLS + MTR predictions, including IKs."""
from torch import nn
import torch
from model.cardiosafe import _init_weights, REGRESSION_HEADS, CLASSIFICATION_HEADS
from model.latent_fusion import CachedConcat

REVISION_CLASSIFICATION_HEADS = CLASSIFICATION_HEADS + ('iks_blocker',)
REVISION_HEADS = REGRESSION_HEADS + REVISION_CLASSIFICATION_HEADS


class RevisionConcat(CachedConcat):
    def __init__(self, chem_dim=6164, encoder_dim=583, chemberta=True, dropout=.3):
        super().__init__(chem_dim=chem_dim, chemberta_dim=encoder_dim,
                         chemberta=chemberta, dropout=dropout, attention='none')
        self.heads['iks_blocker'] = nn.Sequential(nn.Linear(self.shared_dim,128), nn.ReLU(),
                                                nn.Dropout(dropout), nn.Linear(128,1))
        _init_weights(self.heads['iks_blocker'])

    @torch.no_grad()
    def predict(self, x):
        outputs = self(x)
        for i,head in enumerate(REGRESSION_HEADS):
            outputs[head] = outputs[head]*self.reg_std[i]+self.reg_mean[i]
        for head in REVISION_CLASSIFICATION_HEADS:
            outputs[head] = outputs[head].sigmoid()
        return outputs


def build_revision_model(mode='none', slots=4, chemberta=True, chem_dim=6164,
                         dropout=.3, packed_tokens=False, encoder_dim=583):
    if mode!='none' or packed_tokens:
        raise ValueError('The final revision model uses concatenation without token attention')
    return RevisionConcat(chem_dim,encoder_dim,chemberta,dropout)
