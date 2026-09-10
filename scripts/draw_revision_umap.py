"""Label-blind UMAP projections of representative seed42 branch outputs."""
import csv
import json
from pathlib import Path
import umap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.draw_revision_performance import save


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('row_idx');rows=[]
    for split in ['tan70_v1_1','tan60_v1_1']:
        f=np.load(Path('runs_revision_probes')/(split+'_umap_inputs.npz'));ids=f['idx']
        scaffolds=[MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(compounds.loc[i,'smiles'])) for i in ids]
        common=pd.Series(scaffolds).value_counts().head(5).index.tolist()
        classes=np.asarray([common.index(s) if s in common else 5 for s in scaffolds])
        fig,axes=plt.subplots(2,3,figsize=(7.1,5),layout='constrained')
        for j,branch in enumerate(['chemical','chemberta','shared']):
            x=StandardScaler().fit_transform(f[branch])
            saved=Path('reports/paper_revision/tables/phase4_umap_coordinates_both.csv')
            if saved.exists():
                previous=pd.read_csv(saved);previous=previous[(previous.split==split)&(previous.branch==branch)]
                assert np.array_equal(previous.row_idx.to_numpy(),ids)
                xy=previous[['umap1','umap2']].to_numpy()
            else:
                xy=umap.UMAP(n_neighbors=30,min_dist=.1,n_components=2,metric='euclidean',random_state=20260428,n_jobs=1).fit_transform(x)
            y=f['labels'][:,1]
            for value,color,label in [(0,'#b9c7d8','Non-blocker'),(1,'#be532c','Blocker'),(np.nan,'#ddd','Unlabelled')]:
                mask=np.isnan(y) if np.isnan(value) else y==value
                axes[0,j].scatter(*xy[mask].T,s=3,c=color,label=label,rasterized=True,alpha=.7)
            for group in [5,0,1,2,3,4]:
                mask=classes==group
                axes[1,j].scatter(*xy[mask].T,s=3,c=['#c6cbd0' if group==5 else plt.get_cmap('tab10')(group)],rasterized=True,alpha=.7)
            axes[0,j].set_title({'chemical':'Chemical','chemberta':'ChemBERTa-MTR','shared':'Concatenated branches'}[branch])
            for ax in axes[:,j]:ax.set(xticks=[],yticks=[],xlabel='UMAP1',ylabel='UMAP2')
            for i,row in enumerate(ids):rows.append(dict(split=split,seed=42,branch=branch,row_idx=int(row),
                umap1=float(xy[i,0]),umap2=float(xy[i,1]),herg10_label=float(y[i]) if np.isfinite(y[i]) else None,
                scaffold=scaffolds[i],scaffold_group=int(classes[i]),scaffold_group_description='Five most frequent scaffolds;5=all others'))
        axes[0,0].legend(markerscale=2,fontsize=7,loc='lower left')
        axes[1,1].legend(handles=[Line2D([],[],marker='o',ls='',markersize=4,color=plt.get_cmap('tab10')(i) if i<5 else '#c6cbd0',
            label='Scaffold '+str(i+1) if i<5 else 'Other') for i in range(6)],fontsize=7,ncol=2,loc='lower center')
        axes[0,0].set_ylabel('hERG 10 µM labels\nUMAP2');axes[1,0].set_ylabel('Scaffold groups\nUMAP2')
        fig.suptitle(f'{split.replace("_v1_1","")} • seed 42 • label-blind descriptive projections',fontsize=9)
        save(fig,Path('reports/paper_revision/figures')/('phase4_umap_'+split))
    path=Path('reports/paper_revision/tables/phase4_umap_coordinates_both.csv')
    with path.open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    path.with_suffix('.json').write_text(json.dumps(dict(umap_version=umap.__version__,seed=20260428,
        fit='Feature-standardized uniform4096 test sample; labels and scaffold groups used only for post-hoc coloring',
        limitation='Transductive descriptive visualization, not a held-out predictor; axes across branches are not aligned; no query/channel contexts exist in concat'),indent=2))


if __name__=='__main__':main()
