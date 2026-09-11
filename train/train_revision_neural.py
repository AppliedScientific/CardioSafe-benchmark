"""Harmonized D-MPNN and frozen/adapted ChemBERTa reviewer comparisons.

These comparisons use128-row microbatches and16-step accumulation, with local
length bucketing. Frozen_matched isolates encoder adaptation under that same
recipe. They are not replacements for the author's locked final architecture.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from model.revision import RevisionConcat
from model.revision_finetune import RevisionFinetune
from model.revision_dmpnn import RevisionDMPNN
from train.revision_smiles import SmilesTokens
from train.revision_graphs import GraphCache
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from train.train_revision import (HP,LABEL_TO_HEAD,REG_COLS,CLS_COLS,REG_HEAD_IDX,COL_WEIGHT,WRMSE_WEIGHT,
    read_labels,read_split,fit_scalers,fit_task_baselines,scale_labels,noam_lr,atomic_checkpoint)
from train.readout_contract import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['dmpnn','finetune','frozen_matched'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['tan70_v1_1','tan60_v1_1'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--train-limit',type=int);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(int(os.environ.get('TORCH_THREADS','3')));torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed);device=torch.device(a.device)
    folds=read_split(Path('data/splits')/(a.split+'.csv'));train=folds['train']
    labels_path=Path('data/labels/labels_v3.csv');y=read_labels(labels_path,334444)
    scalers,weights=fit_scalers(y,train);baselines=fit_task_baselines(y,train,weights);scaled=scale_labels(y,scalers)
    desc=np.load('data/features/descriptors_20.npy');dm=desc[train].mean(0);ds=desc[train].std(0);ds[ds<1e-8]=1
    cache=Path('data/features')/('revision_'+a.split);scale=json.loads((cache/'manifest.json').read_text())
    provider=build_provider(Path('data/features'),transform_descriptors((desc-dm)/ds,'signed_log'),chemberta_path=cache/'combined.npy')
    # Same original supervised input population as the final MLP. Feature scaling
    # above still uses all global TRAIN rows, as in that model's recorded recipe.
    original_training_rows=len(train);train=train[np.isfinite(y[train]).any(axis=1)]
    ordering=SmilesTokens();graphs=GraphCache() if a.model=='dmpnn' else None
    if a.model=='dmpnn':model=RevisionDMPNN();ordering.lengths=graphs.lengths
    elif a.model=='frozen_matched':model=RevisionConcat()
    else:
        base=next((Path.home()/'.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots').iterdir())
        model=RevisionFinetune(base/'pytorch_model.bin',json.loads((base/'config.json').read_text()),scale)
    for c,(mean,std) in scalers.items():
        i=REG_HEAD_IDX[LABEL_TO_HEAD[c]];model.reg_mean[i]=mean;model.reg_std[i]=std
    model.to(device)
    if a.model=='finetune':
        adapted=list(model.pretrained_parameters());ids={id(p) for p in adapted}
        groups=[dict(params=[p for p in model.parameters() if id(p) not in ids],lr_ratio=1.),dict(params=adapted,lr_ratio=.01)]
    else:groups=[dict(params=list(model.parameters()),lr_ratio=1.)]
    optimizer=torch.optim.AdamW(groups,lr=HP.init_lr,weight_decay=HP.weight_decay)
    bce={c:nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weights[c]],device=device)) for c in CLS_COLS}
    if a.train_limit:train=train[:a.train_limit]
    steps_per_epoch=int(np.ceil(len(train)/2048));step=0;best=float('inf');best_epoch=-1;trajectory=[]
    provenance=dict(model=a.model,split=a.split,seed=a.seed,device=a.device,labels_sha256=sha(labels_path),
        split_sha256=sha(Path('data/splits')/(a.split+'.csv')),source_sha256={str(p):sha(p) for d in ['model','train'] for p in Path(d).glob('*.py')},
        microbatch=128,accumulation=16,effective_batch=2048,maximum_epochs=a.epochs,patience=HP.patience,
        schedule='Locked100-epoch Noam schedule; encoder LR0.01x downstream LR when adapted',
        batch_order='Seeded shuffle, stable local length buckets4096, then shuffled batches',
        loss='Locked pooled normalized eight-head BCE/MSE',selection='Validation primary weighted RMSE',
        train_limit=a.train_limit,smoke=bool(a.train_limit or a.epochs!=100),torch=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),torch_threads=torch.get_num_threads(),
        original_training_rows=original_training_rows,training_rows=len(train))
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))

    def forward(rows):
        if a.model=='dmpnn':return model.forward_graph(graphs.batch(rows,device))
        x=torch.from_numpy(provider[rows]).to(device)
        if a.model=='frozen_matched':return model(x)
        ids,mask=ordering.batch(rows,device);return model.forward_tokens(x,ids,mask)

    @torch.inference_mode()
    def evaluate(indices):
        model.eval();raw=np.full((len(indices),8),np.nan,dtype=np.float32)
        positions={int(v):i for i,v in enumerate(indices)}
        for rows in ordering.ordered_batches(indices,128):
            values=forward(rows);at=[positions[int(i)] for i in rows]
            for col,head in LABEL_TO_HEAD.items():
                value=values[head].cpu().numpy()
                if col in REG_COLS:value=value*scalers[col][1]+scalers[col][0]
                raw[at,col]=value
        assert np.isfinite(raw).all()
        wrmse=0.
        for col in REG_COLS:
            mask=np.isfinite(y[indices,col]);assert mask.sum()>=5
            wrmse+=WRMSE_WEIGHT[LABEL_TO_HEAD[col]]*np.sqrt(np.mean((raw[mask,col]-y[indices[mask],col])**2))
        return raw,float(wrmse)

    for epoch in range(a.epochs):
        started=time.monotonic();model.train();running=0.;batches=ordering.ordered_batches(train,128,rng)
        for start in range(0,len(batches),16):
            window=batches[start:start+16];counts=np.isfinite(scaled[np.concatenate(window)]).sum(0)
            optimizer.zero_grad(set_to_none=True)
            for rows in window:
                values=forward(rows);target=torch.from_numpy(scaled[rows]).to(device);terms=[]
                for col,head in LABEL_TO_HEAD.items():
                    observed=torch.isfinite(target[:,col]);n=int(observed.sum())
                    if not n:continue
                    value=values[head][observed];truth=target[observed,col]
                    loss=nn.functional.mse_loss(value,truth) if col in REG_COLS else bce[col](value,truth)
                    terms.append(COL_WEIGHT[col]*loss*n/counts[col]/baselines[col])
                if terms:
                    loss=torch.stack(terms).sum();loss.backward();running+=float(loss.detach())
            for group in optimizer.param_groups:group['lr']=noam_lr(step,steps_per_epoch)*group['lr_ratio']
            nn.utils.clip_grad_norm_(model.parameters(),HP.grad_clip);optimizer.step();step+=1
        training_seconds=time.monotonic()-started
        raw,score=evaluate(folds['val']);improved=score<best
        record=dict(epoch=epoch,loss=running/max(1,int(np.ceil(len(batches)/16))),val_wrmse=score,seconds=time.monotonic()-started,
            training_seconds=training_seconds,validation_seconds=time.monotonic()-started-training_seconds,best=improved,
            cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30 if a.device=='cuda' else None)
        trajectory.append(record);(a.out/'trajectory.json').write_text(json.dumps(trajectory,indent=2));print(json.dumps(record),flush=True)
        if improved:
            best=score;best_epoch=epoch
            atomic_checkpoint(dict(model_state_dict=model.state_dict(),model=a.model,seed=a.seed,split=a.split,
                reg_scalers=scalers,class_weights=weights,descriptor_scaler_means=dm,descriptor_scaler_stds=ds,
                epoch=epoch,val_wrmse=score,provenance=provenance),a.out/'model.pt')
        elif epoch-best_epoch>=HP.patience:break
    ck=torch.load(a.out/'model.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
    for fold in ['val','test']:
        raw,score=evaluate(folds[fold]);np.savez_compressed(a.out/f'predictions_{fold}_revision.npz',idx=folds[fold],labels=y[folds[fold]],raw=raw)
    (a.out/'result.json').write_text(json.dumps(dict(status='complete',model=a.model,seed=a.seed,split=a.split,
        best_epoch=best_epoch,val_wrmse=best,smoke=provenance['smoke']),indent=2))


if __name__=='__main__':
    from train.revision_gpu_slots import reserve_finetune_slot
    with reserve_finetune_slot():main()
