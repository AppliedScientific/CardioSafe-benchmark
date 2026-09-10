"""Supervised encoder-adaptation comparator; the final released encoder stays frozen."""
import json
from pathlib import Path
import torch
from torch import nn
from transformers import RobertaConfig,RobertaModel
from model.revision import RevisionConcat


class RevisionFinetune(RevisionConcat):
    def __init__(self,weights_path,config,feature_scale):
        # Preserve the downstream initialization associated with the same seed.
        super().__init__(encoder_dim=583)
        with torch.random.fork_rng():
            self.encoder=RobertaModel(RobertaConfig.from_dict(config),add_pooling_layer=False)
            self.mtr_dense=nn.Linear(384,384);self.mtr_out=nn.Linear(384,199)
        weights=torch.load(weights_path,map_location='cpu',weights_only=True)
        state={k.removeprefix('roberta.'):v for k,v in weights.items() if k.startswith('roberta.') and k!='roberta.embeddings.position_ids'}
        self.encoder.load_state_dict(state,strict=True)
        for target,prefix in [(self.mtr_dense,'regression.dense'),(self.mtr_out,'regression.out_proj')]:
            target.load_state_dict({k:weights[prefix+'.'+k] for k in ['weight','bias']},strict=True)
        for name,value in [('property_mean',weights['norm_mean']),('property_std',weights['norm_std']),
            ('feature_mean',feature_scale['mean']),('feature_std',feature_scale['scale'])]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32).clone())

    def encoder_readout(self,ids,mask):
        cls=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state[:,0,:]
        predictions=self.mtr_out(torch.relu(self.mtr_dense(cls)))*self.property_std+self.property_mean
        return torch.cat([cls,(predictions-self.feature_mean)/self.feature_std],1)

    def forward_tokens(self,x,ids,mask):
        readout=self.encoder_readout(ids,mask)
        return super().forward(torch.cat([x[:,:self.chem_dim],readout],1))

    def pretrained_parameters(self):
        yield from self.encoder.parameters();yield from self.mtr_dense.parameters();yield from self.mtr_out.parameters()
