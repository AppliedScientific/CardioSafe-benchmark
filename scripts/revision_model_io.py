"""Strict final-model loader shared by manuscript analyses."""
from pathlib import Path
import numpy as np
import torch
from model.revision import RevisionConcat
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from train.readout_contract import resolve_readout


def load_final(run,device='cpu'):
    ck=torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=False)
    assert ck['revision_version']==1 and ck['attention']=='none' and ck['encoder_dim']==583
    assert ck['labels']=='labels_v3.csv' and ck['y_randomize'] is None
    model=RevisionConcat();model.load_state_dict(ck['model_state_dict'],strict=True);model.to(device).eval()
    desc=np.load('data/features/descriptors_20.npy')
    desc=transform_descriptors((desc-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],ck['descriptor_transform'])
    provider=build_provider(Path('data/features'),desc,chemberta_path=resolve_readout(ck['chemberta_cache'],Path.cwd()))
    return model,provider,ck
