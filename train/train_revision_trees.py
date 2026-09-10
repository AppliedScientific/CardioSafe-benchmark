"""B4: independent endpoint forests or XGBoost on Morgan2048 + descriptors20.

Fixed settings, five seeds; validation is used only for XGBoost early stopping.
This script fits one seed and preserves models and probability predictions.
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from train.train_revision import read_labels,read_split,LABEL_TO_HEAD,REG_COLS
from train.readout_contract import sha


def make_estimator(kind,regression,seed,threads,pos_weight=1.,smoke=False):
    if kind=='rf':
        common=dict(n_estimators=5 if smoke else 300,max_features='sqrt',max_leaf_nodes=4096,
            min_samples_leaf=2,n_jobs=threads,random_state=seed)
        return RandomForestRegressor(**common) if regression else RandomForestClassifier(**common,class_weight='balanced_subsample')
    import xgboost as xgb
    common=dict(n_estimators=5 if smoke else 1000,max_depth=6,learning_rate=.05,subsample=.8,
        colsample_bytree=.8,reg_lambda=1.,tree_method='hist',n_jobs=threads,random_state=seed,
        early_stopping_rounds=50)
    return xgb.XGBRegressor(**common,objective='reg:squarederror',eval_metric='rmse') if regression else xgb.XGBClassifier(
        **common,objective='binary:logistic',eval_metric='logloss',scale_pos_weight=pos_weight)


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['rf','xgb'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'result.json').exists(),'Refusing completed-run overwrite'
    folds=read_split(Path('data/splits')/(a.split+'.csv'))
    fp=np.load('data/features/morgan_fps_2048.npy',mmap_mode='r');d=np.load('data/features/descriptors_20.npy')
    x=np.concatenate([fp,d],axis=1).astype(np.float32);assert x.shape[1]==2068 and np.isfinite(x).all()
    y=read_labels(a.labels,len(x));pred={f:np.full((len(folds[f]),8),np.nan,dtype=np.float32) for f in ['val','test']}
    records=[]
    for col,head in LABEL_TO_HEAD.items():
        train=folds['train'][np.isfinite(y[folds['train'],col])];val=folds['val'][np.isfinite(y[folds['val'],col])]
        if a.smoke:train=train[:min(len(train),256)]
        if len(train)<2 or len(val)<1:
            records.append(dict(head=head,status='insufficient_training_or_validation_labels',train_n=len(train),val_n=len(val)));continue
        target=y[train,col];regression=col in REG_COLS
        if not regression and len(np.unique(target))<2:
            for fold in pred:pred[fold][:,col]=target[0]
            records.append(dict(head=head,status='constant_single_class_training',train_n=len(train),constant=float(target[0])));continue
        pw=float((target==0).sum()/max((target==1).sum(),1)) if not regression else 1.
        model=make_estimator(a.kind,regression,a.seed,a.threads,pw,a.smoke);started=time.time()
        kwargs=dict(eval_set=[(x[val],y[val,col])],verbose=False) if a.kind=='xgb' else {}
        model.fit(x[train],target,**kwargs)
        for fold in pred:
            xx=x[folds[fold]];pred[fold][:,col]=model.predict(xx) if regression else model.predict_proba(xx)[:,list(model.classes_).index(1)]
        model_path=a.out/(head+('.joblib' if a.kind=='rf' else '.json'))
        if a.kind=='rf':joblib.dump(model,model_path,compress=3)
        else:model.save_model(model_path)
        record=dict(head=head,status='complete',train_n=len(train),val_n=len(val),seconds=time.time()-started,
            parameters=model.get_params(),model_sha256=sha(model_path),best_iteration=getattr(model,'best_iteration',None))
        records.append(record);print(head,len(train),round(time.time()-started),flush=True)
        (a.out/'progress.json').write_text(json.dumps(records,indent=2))
    for fold in pred:np.savez_compressed(a.out/f'predictions_{fold}_revision.npz',idx=folds[fold],labels=y[folds[fold]],predictions=pred[fold])
    result=dict(kind=a.kind,seed=a.seed,split=a.split,labels_sha256=sha(a.labels),
        split_sha256=sha(Path('data/splits')/(a.split+'.csv')),smoke=a.smoke,
        feature_layout='Morgan2048 + raw descriptors20; no learned preprocessing',
        classification='probabilities; decision threshold .5',regression='direct unscaled pChEMBL',
        sklearn_version=sklearn.__version__,heads=records)
    if a.kind=='xgb':
        import xgboost;result['xgboost_version']=xgboost.__version__
    (a.out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
