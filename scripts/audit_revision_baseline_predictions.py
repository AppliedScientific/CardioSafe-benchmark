"""Replay baseline exports, verify label identity and audit probability ensembles."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scripts.revision_metrics import EndpointMetrics
from train.train_revision import read_labels,read_split,LABEL_TO_HEAD,REG_COLS,REG_HEAD_IDX,WRMSE_WEIGHT
from train.readout_contract import sha
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from model.revision_baselines import build_baseline_model
from model.revision import RevisionConcat

ARMS={
    'chemical':('dense','chemical'),'static':('dense','static'),
    'single_herg':('dense','single_herg'),'single_nav15':('sparse','single_nav15'),
    'single_cav12':('sparse','single_cav12'),'single_iks':('sparse','single_iks'),
    'ic50':('sparse','ic50'),'herg_10':('dense','herg_10'),
    'herg_25':('dense','herg_25'),'herg_50':('dense','herg_50'),
    'molformer':('encoder','molformer'),'rf':('trees','rf'),'xgb':('trees','xgb'),
    'dmpnn':('neural','dmpnn'),'finetune':('neural','finetune'),
    'frozen_matched':('neural','frozen_matched')}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--available-only',action='store_true');p.add_argument('--tag',default='all');a=p.parse_args()
    torch.set_num_threads(1);descriptors=np.load('data/features/descriptors_20.npy');n=len(descriptors)
    labels={name:read_labels(Path('data/labels')/name,n) for name in ['labels_v3.csv','labels_revision_ic50.csv']}
    folds={s:read_split(Path('data/splits')/(s+'.csv')) for s in ['tan70_v1_1','tan60_v1_1']}
    records=[];points=[];ensemble_checks=[];missing=[];feature_hashes={};graph=None;tokens=None;morgan=None
    for arm in a.arms:
        block,dirname=ARMS[arm]
        for split in folds:
            root=Path('runs_revision_'+block)/dirname/split
            if not (root/'ensemble_test_revision.npz').exists() or not (root/'ensemble_val_revision.npz').exists():
                missing.append(str(root));continue
            truth=labels['labels_revision_ic50.csv' if arm=='ic50' else 'labels_v3.csv'];parts={'val':[],'test':[]}
            for seed in range(42,47):
                run=root/f'seed_{seed}';checkpoint=run/'model.pt';model=None;provider=None
                if block!='trees':
                    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if block=='neural':
                        assert ck['model']==arm and ck['provenance']['smoke'] is False
                        if arm=='dmpnn':
                            from model.revision_dmpnn import RevisionDMPNN
                            from train.revision_graphs import GraphCache
                            model=RevisionDMPNN();graph=graph or GraphCache()
                        elif arm=='finetune':
                            from model.revision_finetune import RevisionFinetune
                            from train.revision_smiles import SmilesTokens
                            base=Path.home()/'.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca'
                            scale=json.loads((Path('data/features')/('revision_'+split)/'manifest.json').read_text())
                            model=RevisionFinetune(base/'pytorch_model.bin',json.loads((base/'config.json').read_text()),scale);tokens=tokens or SmilesTokens()
                        else:model=RevisionConcat()
                        active=list(LABEL_TO_HEAD.values());cache=Path('data/features')/('revision_'+split)/'combined.npy'
                        expected=ck['provenance'];assert expected['labels_sha256']==sha('data/labels/labels_v3.csv')
                        if arm!='dmpnn':
                            desc=transform_descriptors((descriptors-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],'signed_log')
                            provider=build_provider(Path('data/features'),desc,chemberta_path=cache)
                    else:
                        assert ck['revision_version']==1 and ck['y_randomize'] is None
                        assert ck['labels']==('labels_revision_ic50.csv' if arm=='ic50' else 'labels_v3.csv')
                        assert ck['attention']==('static' if arm=='static' else 'none')
                        assert ck['chemberta']==(arm!='chemical')
                        if arm=='molformer':assert ck['encoder_dim']==768
                        if arm.startswith('herg_'):
                            assert ck['supervision']['herg_fraction']==int(arm.split('_')[1])/100
                        if arm.startswith('single_'):
                            assert ck['supervision']['channel']==arm.removeprefix('single_')
                        model=build_baseline_model(mode=ck['attention'],encoder_dim=ck['encoder_dim'],chemberta=ck['chemberta'])
                        active=ck['active_heads']
                        for h in list(model.heads):
                            if h not in active:del model.heads[h]
                        cache=Path(ck['chemberta_cache']['path'])
                        key=str(cache)
                        if key not in feature_hashes:feature_hashes[key]=sha(cache)
                        assert feature_hashes[key]==ck['chemberta_cache']['sha256']
                        desc=transform_descriptors((descriptors-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],ck['descriptor_transform'])
                        provider=build_provider(Path('data/features'),desc,chemberta_path=cache)
                    assert ck['seed']==seed and ck['split']==split
                    model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
                    params=sum(p.numel() for p in model.parameters());model_hash=sha(checkpoint)
                else:
                    result=json.loads((run/'result.json').read_text());assert result['smoke'] is False
                    assert result['seed']==seed and result['split']==split and result['labels_sha256']==sha('data/labels/labels_v3.csv')
                    active=list(LABEL_TO_HEAD.values());params=None;model_hash=None
                    if morgan is None:morgan=np.load('data/features/morgan_fps_2048.npy',mmap_mode='r')
                columns=[c for c,h in LABEL_TO_HEAD.items() if h in active]
                for fold in parts:
                    f=np.load(run/f'predictions_{fold}_revision.npz');idx=f['idx']
                    np.testing.assert_array_equal(idx,folds[split][fold]);np.testing.assert_array_equal(f['labels'],truth[idx])
                    chosen=np.unique(np.r_[np.linspace(0,len(idx)-1,128,dtype=int),np.flatnonzero(np.isfinite(truth[idx,7]))]);ids=idx[chosen]
                    actual=np.full((len(ids),8),np.nan,dtype=np.float32)
                    if block=='trees':
                        import joblib
                        x=np.concatenate([morgan[ids],descriptors[ids]],1).astype(np.float32)
                        for c,h in LABEL_TO_HEAD.items():
                            meta=next(r for r in result['heads'] if r['head']==h)
                            if meta['status']=='constant_single_class_training':actual[:,c]=meta['constant'];continue
                            assert meta['status']=='complete'
                            path=run/(h+('.joblib' if arm=='rf' else '.json'));assert sha(path)==meta['model_sha256']
                            if arm=='rf':estimator=joblib.load(path)
                            else:
                                import xgboost as xgb
                                estimator=xgb.XGBRegressor() if c in REG_COLS else xgb.XGBClassifier();estimator.load_model(path)
                            actual[:,c]=estimator.predict(x) if c in REG_COLS else estimator.predict_proba(x)[:,list(estimator.classes_).index(1)]
                            del estimator
                        stored=f['predictions'];prob=stored.copy()
                    else:
                        for start in range(0,len(ids),64):
                            rows=ids[start:start+64]
                            if arm=='dmpnn':out=model.forward_graph(graph.batch(rows,'cpu'))
                            elif arm=='finetune':
                                t,m=tokens.batch(rows,'cpu');out=model.forward_tokens(torch.from_numpy(provider[rows]),t,m)
                            else:out=model(torch.from_numpy(provider[rows]))
                            for c,h in LABEL_TO_HEAD.items():
                                if h not in out:continue
                                value=out[h]
                                if c in REG_COLS:value=value*model.reg_std[REG_HEAD_IDX[h]]+model.reg_mean[REG_HEAD_IDX[h]]
                                actual[start:start+len(rows),c]=value.numpy()
                        stored=f['raw'];prob=stored.copy()
                        for c in LABEL_TO_HEAD:
                            if c not in REG_COLS:prob[:,c]=expit(prob[:,c])
                    assert np.isfinite(prob[:,columns]).all()
                    np.testing.assert_allclose(actual,stored[chosen],atol=2e-4,rtol=2e-5,equal_nan=True)
                    parts[fold].append(prob)
                    records.append(dict(arm=arm,split=split,seed=seed,fold=fold,rows=len(ids),parameters=params,
                        max_absolute_error=float(np.nanmax(np.abs(actual-stored[chosen]))),checkpoint_sha256=model_hash))
                    for c in columns:
                        mask=np.isfinite(truth[idx,c]);yy=truth[idx,c][mask];pp=prob[:,c][mask]
                        if len(yy)<5:continue
                        for metric,value in EndpointMetrics(yy,pp,c in REG_COLS)().items():
                            points.append(dict(arm=arm,split=split,seed=seed,fold=fold,head=LABEL_TO_HEAD[c],metric=metric,estimate=value,n=len(yy),truth='strict_ic50' if arm=='ic50' else 'v3'))
                del model,provider
            for fold,arrays in parts.items():
                f=np.load(root/f'ensemble_{fold}_revision.npz');np.testing.assert_array_equal(f['idx'],folds[split][fold]);np.testing.assert_array_equal(f['labels'],truth[f['idx']])
                expected=np.mean(arrays,axis=0);np.testing.assert_allclose(f['predictions'],expected,atol=2e-7,rtol=0,equal_nan=True)
                ensemble_checks.append(dict(arm=arm,split=split,fold=fold,max_absolute_error=float(np.nanmax(np.abs(f['predictions']-expected)))))
            print(arm,split,'passed',flush=True)
    assert a.available_only or not missing,missing
    root=Path('reports/paper_revision');pd.DataFrame(points).to_csv(root/f'tables/phase3_baseline_per_seed_{a.tag}.csv',index=False)
    (root/f'baseline_inference_audit_{a.tag}.json').write_text(json.dumps(dict(status='passed',missing=missing,records=records,ensemble_checks=ensemble_checks,feature_sha256=feature_hashes,
        sampling='128 evenly spaced positions plus every observed IKs molecule in each fold',tolerance='2e-4 absolute /2e-5 relative across CPU/CUDA; ensemble2e-7'),indent=2))

if __name__=='__main__':main()
