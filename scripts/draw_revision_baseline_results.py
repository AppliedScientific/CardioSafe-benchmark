"""Journal-width baseline and transfer figures from complete paired metric tables."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.draw_revision_performance import save

NAMES={'final':'Final concat','chemical':'Chemical only','static':'Static attention','single_channel':'Single channel',
       'rf':'Random forest','xgb':'XGBoost','dmpnn':'D-MPNN','molformer':'MolFormer',
       'frozen_matched':'Frozen MTR, matched','finetune':'Fine-tuned MTR'}
HEADS={'herg_blocker_10um':'hERG 10 µM','herg_blocker_1um':'hERG 1 µM',
       'nav15_blocker':'NaV1.5','cav12_blocker':'CaV1.2','iks_blocker':'IKs (exploratory)'}


def forest(axes,table,metric,comparison,models):
    for j,(head,title) in enumerate(HEADS.items()):
        ax=axes[j];d=table[(table['head']==head)&(table.metric==metric)&(table.comparison==comparison)]
        n=int(d.n.iloc[0]);ax.set_title(f'{title}\nn={n:,}',fontsize=7)
        for i,name in enumerate(models):
            r=d[d.model==name].iloc[0];color='#235aa6' if name=='final' else '#5b6370'
            if np.isfinite(r.estimate):
                ax.hlines(i,r.ci_lower,r.ci_upper,color=color,lw=1)
                ax.plot(r.estimate,i,'o',ms=3.5,color=color,mfc='white' if head=='iks_blocker' else color)
        ax.set(ylim=(len(models)-.5,-.5),yticks=range(len(models)),yticklabels=[NAMES[m] for m in models] if j==0 else [])
        ax.grid(axis='x',alpha=.15)
        ax.axvline(.5 if comparison=='absolute' else 0,color='.6',ls='--',lw=.8)
        ax.tick_params(axis='both',labelsize=7)
        if comparison=='absolute':ax.set(xlim=(-.025,1.025),xticks=[0,.5,1],xlabel='AUROC')
        else:ax.set_xlabel('ΔAUROC')
    if comparison!='absolute':
        lo=min(a.get_xlim()[0] for a in axes[:-1]);hi=max(a.get_xlim()[1] for a in axes[:-1])
        for a in axes[:-1]:a.set_xlim(lo,hi)


def main():
    root=Path('reports/paper_revision');protocol=pd.read_json(root/'tables/phase3_baseline_protocol.json',typ='series')
    assert bool(protocol['complete']),'Do not present an incomplete matrix as final'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    for split in ['tan70_v1_1','tan60_v1_1']:
        table=pd.read_csv(root/f'tables/phase3_baselines_{split}.csv');models=list(NAMES)
        fig,axes=plt.subplots(1,5,figsize=(7.1,4.8),layout='constrained');forest(axes,table,'auc_roc','absolute',models)
        fig.suptitle(f'{split[:5]} • Five-seed ensembles • 95% molecule-bootstrap intervals\nIKs has one TEST positive; its conditional intervals do not establish generalization',fontsize=8)
        save(fig,root/f'figures/phase3_baseline_auroc_{split}')
        fig,axes=plt.subplots(1,5,figsize=(7.1,4.8),layout='constrained');forest(axes,table,'auc_roc','delta_vs_final',models[1:])
        fig.suptitle(f'{split[:5]} • Baseline minus final concat • Paired 95% intervals\nPositive differences favor the baseline; comparisons are unadjusted for multiplicity',fontsize=8)
        save(fig,root/f'figures/phase3_baseline_deltas_{split}')
        fig,axes=plt.subplots(2,2,figsize=(7.1,4.8),layout='constrained')
        for ax,(head,metric,title) in zip(axes.flat,[('nav15_blocker','auc_roc','NaV1.5 AUROC'),('cav12_blocker','auc_roc','CaV1.2 AUROC'),('nav15_pchembl','rmse','NaV1.5 RMSE'),('cav12_pchembl','rmse','CaV1.2 RMSE')]):
            selected=[]
            for name in ['herg_10','herg_25','herg_50','final']:
                r=table[(table.model==name)&(table['head']==head)&(table.metric==metric)&(table.comparison=='absolute')].iloc[0];selected.append(r)
            x=np.array([10,25,50,100]);y=np.array([r.estimate for r in selected]);lo=np.array([r.ci_lower for r in selected]);hi=np.array([r.ci_upper for r in selected])
            # Percentile intervals need not bracket their point estimates.
            ax.vlines(x,lo,hi,color='#235aa6',lw=1);ax.plot(x,y,'o-',color='#235aa6',ms=3.5,lw=1)
            ax.set(title=title,xlabel='Retained hERG TRAIN labels (%)',xticks=x,ylabel='AUROC' if metric=='auc_roc' else 'pChEMBL RMSE');ax.grid(alpha=.15)
        fig.suptitle(f'{split[:5]} • hERG label-volume intervention • Five-seed ensembles\nOriginal TRAIN input population fixed; conditional 95% molecule-bootstrap intervals',fontsize=8)
        save(fig,root/f'figures/phase3_transfer_sweep_{split}')

if __name__=='__main__':main()
