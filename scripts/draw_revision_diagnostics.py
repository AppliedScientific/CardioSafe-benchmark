"""Scientific figures from verified final-model tables; no refitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.draw_revision_performance import save
from train.readout_contract import sha

TABLE=Path('reports/paper_revision/tables')
FIG=Path('reports/paper_revision/figures')
HEADS=['herg_blocker_10um','herg_blocker_1um','nav15_blocker','cav12_blocker','iks_blocker']
NAMES=['hERG 10 µM','hERG 1 µM','NaV1.5','CaV1.2','IKs (exploratory)']
SPLITS=['tan70_v1_1','tan60_v1_1']


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    sources=[]
    for split in SPLITS:
        path=TABLE/f'phase4_domain_{split}.csv';d=pd.read_csv(path);sources.append(path)
        fig,axes=plt.subplots(3,5,figsize=(7.1,6),layout='constrained')
        bins=['similarity_0.0_0.3','similarity_0.3_0.5','similarity_0.5_0.7']
        for j,(head,name) in enumerate(zip(HEADS,NAMES)):
            for i,(metric,label) in enumerate([('auc_roc','AUROC'),('mcc','MCC'),('auprc','Average precision')]):
                ax=axes[i,j];g=d[(d['head']==head)&(d.metric==metric)].set_index('cohort').reindex(bins)
                v=g.estimate.to_numpy();lo=g.ci_lower.to_numpy();hi=g.ci_upper.to_numpy()
                ax.vlines(np.arange(3),lo,hi,color='#235aa6',lw=1.1)
                sparse=np.minimum(g.prevalence*g.n,(1-g.prevalence)*g.n).to_numpy()<5
                ax.scatter(np.arange(3)[~sparse],v[~sparse],color='#235aa6',s=20,zorder=3)
                ax.scatter(np.arange(3)[sparse],v[sparse],facecolors='white',edgecolors='#235aa6',s=20,zorder=3)
                ax.axhline(.5 if metric=='auc_roc' else 0,color='.7',ls='--',lw=.7)
                if metric=='auprc':ax.scatter(np.arange(3),g.prevalence,color='#ae562b',s=12,marker='_')
                ax.set(xticks=range(3),xticklabels=['<.3','.3–.5','.5–.7' if split==SPLITS[0] else '.5–.6'],ylim=(-.45 if metric=='mcc' else -.03,1.07))
                ax.tick_params(labelsize=7)
                ax.grid(axis='y',alpha=.15)
                if i==0:
                    counts='/'.join('—' if pd.isna(n) else str(int(n)) for n in g.n)
                    ax.set_title(name+'\nn='+counts,fontsize=8)
                if j==0:ax.set_ylabel(label)
                if i==2:ax.set_xlabel('Train similarity',fontsize=7)
        fig.suptitle(f'{split[:5]} test • applicability domain • 95% molecule-bootstrap intervals\nOrange: prevalence. Hollow: fewer than 5 minority-class labels.',fontsize=9)
        save(fig,FIG/('phase4_applicability_'+split))
        path=TABLE/f'phase4_comparators_{split}.csv';d=pd.read_csv(path);sources.append(path)
        fig,ax=plt.subplots(figsize=(7.1,3.6),layout='constrained')
        g=d[(d.metric=='auc_roc')&(d.cohort=='all_models_fp99_clean')&(d.comparison!='absolute')]
        if not len(g):
            print('Available comparator cohorts:',sorted(d.cohort.unique()));raise ValueError('Common-clean cohort missing')
        labels=[];position=0
        for head,name in zip([HEADS[0],HEADS[2],HEADS[3]],[NAMES[0],NAMES[2],NAMES[3]]):
            rows=g[g['head']==head]
            for _,r in rows.iterrows():
                assert r.comparison=='delta_vs_final'
                ax.hlines(position,-r.ci_upper,-r.ci_lower,color='#235aa6',lw=1.5);ax.scatter(-r.estimate,position,color='#235aa6',s=25)
                labels.append(name+' vs '+{'ctoxpred2':'CToxPred2 DNN','cardiogenai':'CardioGenAI'}[r.model]+f' (n={int(r.n):,})');position+=1
        ax.axvline(0,color='.55',ls='--',lw=.8)
        ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='ΔAUROC: final − comparator',xlim=(-.2,.45),ylim=(len(labels)-.5,-.5))
        ax.grid(axis='x',alpha=.15)
        fig.suptitle(f'{split[:5]} • paired 95% intervals on a common test cohort\nMorgan similarity <0.99 to both public training-reference pools',fontsize=9)
        save(fig,FIG/('phase4_comparator_deltas_'+split))
        path=TABLE/'phase4_attribution_importance_both.csv';d=pd.read_csv(path);sources.append(path)
        fig,axes=plt.subplots(2,3,figsize=(7.1,5.4),layout='constrained');axes.flat[-1].axis('off')
        for ax,head,name in zip(axes.flat,HEADS,NAMES):
            g=d[(d.split==split)&(d['head']==head)&(d.kind=='descriptor')].nlargest(7,'mean_abs_attribution').iloc[::-1]
            ax.barh(g.feature,g.mean_abs_attribution,color='#6c5193');ax.set_title(name,fontsize=9);ax.set_xlabel('Mean |IG| (logit units)');ax.grid(axis='x',alpha=.15)
        fig.suptitle(f'{split[:5]} • descriptor IG conditional on the observed MTR branch\nUp to 128 test molecules per class; five-model mean; no causal interpretation',fontsize=9)
        save(fig,FIG/('phase4_descriptor_ig_'+split))
    (FIG/'phase4_diagnostic_sources.json').write_text(json.dumps({str(p):sha(p) for p in set(sources)},indent=2))


if __name__=='__main__':main()
