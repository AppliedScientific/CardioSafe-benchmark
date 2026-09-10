"""Export final-model ROC/PR and regression figures as SVG/PDF/600dpi PNG."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from train.train_revision import LABEL_TO_HEAD,REG_COLS,CLS_COLS
from train.readout_contract import sha

NAMES={0:'hERG potency',1:'hERG 10 µM',2:'hERG 1 µM',3:'NaV1.5 potency',4:'NaV1.5',5:'CaV1.2 potency',6:'CaV1.2',7:'IKs (exploratory)'}


def save(fig,path):
    for suffix in ['svg','pdf','png']:fig.savefig(path.with_suffix('.'+suffix),dpi=600,bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    root=Path('reports/paper_revision/figures');hashes={}
    for split in ['tan70_v1_1','tan60_v1_1']:
        source=Path('runs_revision_v3/final')/split/'ensemble_test_revision.npz';f=np.load(source);hashes[str(source)]=sha(source)
        y,p=f['labels'],f['predictions'];fig,axes=plt.subplots(2,5,figsize=(7.1,4.2),layout='constrained')
        for j,col in enumerate(CLS_COLS):
            mask=np.isfinite(y[:,col]);truth=y[mask,col];prob=p[mask,col];n=len(truth);pos=int(truth.sum())
            roc,pr=axes[:,j];roc.plot([0,1],[0,1],color='.75',ls='--',lw=.8)
            if len(np.unique(truth))==2:
                x,z,_=roc_curve(truth,prob);roc.plot(x,z,lw=1.5,color='#235aa6')
                precision,recall,_=precision_recall_curve(truth,prob);pr.step(recall,precision,where='post',lw=1.5,color='#235aa6')
                roc.text(.97,.05,f'AUROC {roc_auc_score(truth,prob):.3f}',ha='right',transform=roc.transAxes,fontsize=8)
                pr.text(.97,.96,f'AP {average_precision_score(truth,prob):.3f}',ha='right',va='top',transform=pr.transAxes,fontsize=8)
            pr.axhline(truth.mean(),color='.55',ls='--',lw=.8)
            roc.set_title(f'{NAMES[col]}\nn={n:,}\npositive={pos:,}',fontsize=7)
            for ax in [roc,pr]:ax.set(xlim=(0,1),ylim=(0,1));ax.grid(alpha=.15)
            roc.set_xlabel('False-positive rate');pr.set_xlabel('Recall')
            if j==0:roc.set_ylabel('Sensitivity');pr.set_ylabel('Precision')
            if pos<5:pr.text(.04,.45,'Only one positive\nGeneralization\nis uncertain',transform=pr.transAxes,fontsize=7,color='#8c3c16')
        fig.suptitle(f'Final model • {split.replace("_v1_1", "")} test • five-seed probability ensemble',fontsize=9)
        save(fig,root/('phase4_roc_pr_'+split))
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.6),layout='constrained')
        for ax,col in zip(axes,REG_COLS):
            mask=np.isfinite(y[:,col]);truth=y[mask,col];pred=p[mask,col]
            density=ax.hexbin(truth,pred,gridsize=35,mincnt=1,cmap='Blues',bins='log',rasterized=True)
            fig.colorbar(density,ax=ax,shrink=.7,label='Molecules (log scale)')
            lo=float(min(truth.min(),pred.min()));hi=float(max(truth.max(),pred.max()));ax.plot([lo,hi],[lo,hi],color='.6',ls='--',lw=.8)
            ax.set(xlabel='Observed pChEMBL',ylabel='Predicted pChEMBL',title=f'{NAMES[col]} • n={len(truth):,}')
            ax.text(.04,.95,f'RMSE {np.sqrt(np.mean((truth-pred)**2)):.3f}',va='top',transform=ax.transAxes,fontsize=8)
        save(fig,root/('phase4_regression_'+split))
    (root/'phase4_performance_figure_sources.json').write_text(json.dumps(dict(input_sha256=hashes,
        note='Curves are fixed-ensemble estimates; use accompanying tables for conditional bootstrap intervals. AP denotes average precision.'),indent=2))


if __name__=='__main__':main()
