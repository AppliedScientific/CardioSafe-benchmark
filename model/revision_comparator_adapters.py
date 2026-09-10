"""Explicit, audited compatibility adapters for pinned comparator artifacts.

Upstream source is kept unmodified. Every active model tensor must load.
The adapters do not fit any parameters or choose settings from test labels.
"""
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import numpy as np
import torch

ASSETS=Path(__file__).resolve().parents[1]/'assets/revision_comparators'


def load_cga(device='cpu'):
    base=ASSETS/'CardioGenAI';sys.path.insert(0,str(base))
    from src.Transformer import Transformer_Model,Transformer_Feature_Extractor
    from src.Discriminator import Discriminator_Model
    vocab=json.loads((base/'vocabulary.json').read_text())
    extractor=Transformer_Feature_Extractor.__new__(Transformer_Feature_Extractor)
    torch.nn.Module.__init__(extractor);extractor.device=torch.device(device)
    extractor.dataset=SimpleNamespace(stoi={s:i for i,s in enumerate(vocab['vocab'])},
        smiles_regex=re.compile(vocab['regex']),block_size=vocab['block_size'])
    extractor.model=Transformer_Model(mode='Bidirectional',vocab_size=len(vocab['vocab']),block_size=vocab['block_size'],extract_features=True)
    ck=torch.load(base/'model_parameters/transformer_model_parameters/Bidirectional_Transformer_parameters.pt',weights_only=True,map_location='cpu')
    extractor.model.load_state_dict(ck['model_state_dict'],strict=True);extractor.model.to(device).eval()
    models={};audits={}
    for channel,name in [('herg','hERG'),('nav15','Nav'),('cav12','Cav')]:
        model=Discriminator_Model(device=device).eval()
        path=base/f'model_parameters/discriminative_model_parameters/{name}_Classification_parameters.pt'
        state=torch.load(path,weights_only=True,map_location='cpu')['model_state_dict'];state=dict(state);aliases=[]
        for layer in ['gat1','gat2']:
            source=state.pop(layer+'.lin_src.weight');destination=state.pop(layer+'.lin_dst.weight')
            assert torch.equal(source,destination),'Cannot combine distinct GAT projections'
            state[layer+'.lin.weight']=source;aliases.append(layer+': identical lin_src/lin_dst -> lin')
        extra=set(state)-set(model.state_dict())
        # The upstream artifact stores heads for six unused feature-subset
        # variants. They do not participate in the full graph+FP+TFV model.
        variants=('gat','fp','tfv','gat_fp','gat_tfv','fp_tfv')
        allowed=tuple(prefix+variant+'.' for variant in variants for prefix in ['combined_linear1_','combined_linear2_','bn_combined_'])
        assert all(k.startswith(allowed) for k in extra),extra
        active={k:v for k,v in state.items() if k not in extra}
        model.load_state_dict(active,strict=True);model.to(device).eval();models[channel]=model
        audits[channel]=dict(gat_aliases=aliases,unused_feature_subset_tensors=sorted(extra),active_tensors=len(active))
    return extractor,models,audits


def cga_token_batch(extractor,smiles):
    ds=extractor.dataset;rows=[]
    for smi in smiles:
        tokens=ds.smiles_regex.findall('[CLS]'+smi.strip()+'[EOS]')
        if len(tokens)>ds.block_size:raise ValueError('SMILES exceeds author positional limit')
        tokens+=['<pad>']*(ds.block_size-len(tokens));rows.append([ds.stoi[t] for t in tokens])
    return torch.tensor(rows,dtype=torch.long,device=extractor.device)


def fixed_descriptor_transform(pipeline,frame):
    """Replay the saved sklearn1.3 imputer/scaler/selectors using their arrays.

    Current sklearn rejects the old custom selector as unfitted. No refit is
    permissible: all-missing columns, scaling and selection stay pretrained.
    """
    steps=pipeline.named_steps;imputer=steps['imputer'];scaler=steps['scaler']
    assert list(frame.columns)==list(imputer.feature_names_in_)
    x=frame.to_numpy(dtype=np.float64);assert not np.isinf(x).any()
    stats=np.asarray(imputer.statistics_);keep=np.isfinite(stats)
    x=np.where(np.isnan(x),stats,x)[:,keep]
    x=(x-scaler.mean_)/scaler.scale_
    var=steps['varr_selector'];x=x[:,var.variances_>var.threshold]
    x=x[:,steps['corr_selector'].mask]
    assert np.isfinite(x).all();return x
