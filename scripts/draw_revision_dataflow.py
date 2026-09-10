"""Source and global-split figure from audited v3 counts (no fitted-model data)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


def main():
    root=Path('reports/paper_revision');coverage=pd.read_csv(root/'tables/phase2_channel_coverage_both.csv')
    audit=json.loads((root/'notes/phase2_split_provenance.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.1,7.9));fig.subplots_adjust(left=.025,right=.975,bottom=.02,top=.98)
    ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    def box(x,y,w,h,label,color='#eef3fa',size=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.35,rounding_size=1',
            fc=color,ec='#36536a',lw=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
    def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=.8,color='#36536a'))
    ax.text(50,98,'One compound pool, one assignment per split',ha='center',va='top',fontsize=13,weight='bold')
    source=[('ChEMBL 36\n22,465 released compounds',1),('hERGCentral screen\n306,671 released compounds',35),('CToxPred curated exports\n24,535 released compounds',69)]
    for label,x in source:box(x,81,30,11,label);arrow(x+15,81,50,76)
    box(12,65,76,11,'334,444 unique standardized compounds\nSource memberships overlap; all compounds accounted for\nFrozen global partitions; channel labels applied afterward','#e9f4eb',8.5)
    for x,split in [(1,'tan70'),(52,'tan60')]:
        counts=audit['splits'][split]['final_counts'];arrow(50,65,x+23,56)
        box(x,28,47,29,'',color='#f7f9fb')
        ax.text(x+23.5,54,split+' v1.1',ha='center',fontsize=11,weight='bold')
        ax.text(x+23.5,49,' / '.join(f'{counts[f]:,}' for f in ['train','val','test']),ha='center',fontsize=9)
        ax.text(x+23.5,46,'TRAIN / VALIDATION / TEST',ha='center',fontsize=6.8,color='#45596a')
        d=coverage[coverage.split==split].pivot(index='channel',columns='fold',values='n_unique_labelled_compounds')
        ax.text(x+2,42,'Labelled channel',fontsize=7,weight='bold');ax.text(x+45,42,'Train      Val      Test',ha='right',fontsize=7,weight='bold')
        for j,ch in enumerate(['hERG','NaV1.5','CaV1.2','IKs']):
            y=38.5-j*2.7;ax.text(x+2,y,ch,fontsize=7.5)
            for xx,f in [(x+25,'train'),(x+35,'val'),(x+45,'test')]:ax.text(xx,y,f'{int(d.loc[ch,f]):,}',ha='right',fontsize=7.2)
        ax.text(x+23.5,24,'Unassigned: '+str(audit['splits'][split]['final_unassigned']),ha='center',fontsize=8)
    box(1,4,98,15,'Zero cross-fold duplicate standardized identities. No splits regenerated here.\n'
        'Forced validation: terfenadine/fexofenadine families and v1.1 corrections.\n'
        'tan60 omits one odd-pool remainder; the claimed 753 exclusions are unsupported.\n'
        'Historical cliff fine-tuning set quarantined after the structure/potency audit.',color='#fff6e6',size=8)
    out=root/'figures';out.mkdir(exist_ok=True)
    for ext in ['svg','pdf','png']:fig.savefig(out/f'phase2_source_split_flow.{ext}',dpi=600)
    plt.close(fig)


if __name__=='__main__':main()
