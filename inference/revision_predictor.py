"""Locked v3 eight-head ensemble from raw, aligned chemical and MTR features.

This entry point is separate from the historical seven-head/mean-logit API.
It does not invent missing descriptors or download a replacement encoder.
"""
import json
from pathlib import Path
import numpy as np
import torch
from model.revision import RevisionConcat,REVISION_HEADS
from train.fusion_preprocessing import transform_descriptors


class RevisionEnsemble:
    def __init__(self, checkpoints, feature_manifest, device='cpu'):
        self.device=torch.device(device);self.models=[];self.scalers=[];self.checkpoints=list(map(Path,checkpoints))
        manifest=json.loads(Path(feature_manifest).read_text())
        if manifest.get('status')!='complete' or manifest.get('fit_fold')!='train':
            raise ValueError('A completed TRAIN-only revision feature manifest is required')
        self.prediction_mean=np.asarray(manifest['mean'],dtype=np.float64)
        self.prediction_scale=np.asarray(manifest['scale'],dtype=np.float64)
        if self.prediction_mean.shape!=(199,) or self.prediction_scale.shape!=(199,) or not np.all(self.prediction_scale>0) or not np.isfinite(self.prediction_mean).all() or not np.isfinite(self.prediction_scale).all():
            raise ValueError('Invalid MTR199 feature scaling')
        seeds=[]
        for path in self.checkpoints:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if not (ck.get('revision_version')==1 and ck.get('attention')=='none' and ck.get('encoder_dim')==583
                    and ck.get('chemberta') and ck.get('labels')=='labels_v3.csv' and ck.get('y_randomize') is None):
                raise ValueError('Expected a final v3 concat checkpoint with frozen CLS384 + MTR199')
            if ck['split']!=manifest['contract']['split'] or ck['chemberta_cache']['sha256']!=manifest['output_sha256']['combined']:
                raise ValueError('Checkpoint and MTR feature preprocessing contract disagree')
            if set(ck.get('active_heads',[]))!=set(REVISION_HEADS):raise ValueError('All eight final heads are required')
            if ck.get('cav_label_fraction',1.)!=1. or ck.get('supervision',{}).get('herg_fraction')!=1.:
                raise ValueError('Full-label final checkpoints are required')
            model=RevisionConcat();model.load_state_dict(ck['model_state_dict'],strict=True);model.to(self.device).eval()
            mean=np.asarray(ck['descriptor_scaler_means']);std=np.asarray(ck['descriptor_scaler_stds'])
            if mean.shape!=(20,) or std.shape!=(20,) or not np.all(std>0):raise ValueError('Invalid descriptor scaling')
            self.models.append(model);self.scalers.append((mean,std,ck['descriptor_transform']));seeds.append(ck['seed'])
        if sorted(seeds)!=list(range(42,47)):raise ValueError('Final ensemble requires seeds42–46 exactly once')
        self.split=manifest['contract']['split']

    @torch.inference_mode()
    def predict_features(self,chemical,cls,mtr_predictions,batch_size=1024):
        """Return five-model mean sigmoid scores and unscaled pChEMBL.

        chemical:N×6164 (three binary fingerprints then RAW20 descriptors);
        cls:N×384 FINAL-layer first-token coordinates;
        mtr_predictions:N×199 pretrained-head outputs AFTER authors' denormalization,
        BEFORE downstream TRAIN feature standardization. Arrays are not mutated.
        """
        chemical,cls,mtr_predictions=map(np.asarray,(chemical,cls,mtr_predictions))
        n=len(chemical)
        for value,width in [(chemical,6164),(cls,384),(mtr_predictions,199)]:
            if value.shape!=(n,width) or not np.isfinite(value).all():raise ValueError(f'Expected finite aligned ({n},{width}) feature array')
        if not isinstance(batch_size,int) or batch_size<1:raise ValueError('batch_size must be a positive integer')
        result={head:np.empty(n,dtype=np.float32) for head in REVISION_HEADS}
        for start in range(0,n,batch_size):
            stop=min(start+batch_size,n)
            x=np.concatenate([chemical[start:stop],cls[start:stop],
                (mtr_predictions[start:stop]-self.prediction_mean)/self.prediction_scale],axis=1).astype(np.float32)
            predictions=[]
            for model,(mean,std,transform) in zip(self.models,self.scalers):
                xx=x.copy();xx[:,6144:6164]=transform_descriptors((xx[:,6144:6164]-mean)/std,transform)
                values=model.predict(torch.from_numpy(xx).to(self.device))
                predictions.append({h:v.cpu().numpy() for h,v in values.items()})
            for head in result:result[head][start:stop]=np.mean([p[head] for p in predictions],axis=0)
        return result
