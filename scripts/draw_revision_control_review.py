"""Show the unresolved real-label ranking without replacing ensemble AUROCs by means."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.draw_revision_performance import save


def main():
    root=Path('reports/paper_revision');fit=pd.read_csv(root/'tables/phase1_control_fit_bootstrap_both.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.1,3.2),layout='constrained');labels=[]
    for i,(arm,split) in enumerate([(a,s) for a in ['yrandom','yrandom_rows'] for s in ['tan70_v1_1','tan60_v1_1']]):
        d=pd.read_csv(root/f'tables/phase1_independent_{arm}_v3_{split}_test.csv');r=d[(d['head']=='herg_blocker_10um')&(d.metric=='auc_roc')].iloc[0];f=fit[(fit.model==arm)&(fit.split==split)&(fit['head']=='herg_blocker_10um')].iloc[0]
        for off,q,color,label in [(-.13,r,'#235aa6','Molecule bootstrap'),(.13,f,'#b26231','Fitted-model resampling')]:
            ax.hlines(i+off,q.ci_lower,q.ci_upper,color=color,lw=1.4);ax.scatter(q.estimate,i+off,color=color,s=22,label=label if i==0 else None)
        labels.append(('Observed cells' if arm=='yrandom' else 'Whole label rows')+' • '+split[:5])
    ax.axvline(.5,color='.45',ls='--',lw=.9);ax.set(yticks=range(4),yticklabels=labels,xlim=(.25,.75),ylim=(3.5,-.95),xlabel='hERG 10 µM AUROC against untouched real TEST labels');ax.legend(loc='upper center',fontsize=7,ncol=2)
    fig.suptitle('Randomized-label controls: real-label ranking remains\n95% conditional intervals use different uncertainty sources; five-fit resampling is descriptive',fontsize=9)
    save(fig,root/'figures/phase1_control_review')


if __name__=='__main__':main()
