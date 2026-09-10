"""Figure1: final eight-output architecture with explicit frozen and trained paths."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def main():
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.1,5));ax.set(xlim=(0,1200),ylim=(730,0));ax.axis('off')
    palettes={'chem':('#e4edff','#355bb6'),'cb':('#efe8ff','#7951b0'),
              'join':('#fff0dc','#aa5525'),'cls':('#fff3c3','#aa7725'),
              'reg':('#ffdfa0','#aa7725'),'iks':('#eeeeee','#777777')}
    def box(x,y,w,h,kind,text,size=7.5):
        fill,edge=palettes[kind]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=4',
                                   facecolor=fill,edgecolor=edge,lw=1.2,zorder=3))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.4,zorder=4)
    def arrow(x,y,xx,yy,rad=0):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='->',mutation_scale=9,
                    connectionstyle=f'arc3,rad={rad}',color='#555555',lw=1.2,zorder=2))
    ax.text(600,28,'CardioSafe: chemical features + ChemBERTa representation and predictions',
            ha='center',fontsize=9,weight='bold')
    ax.text(600,55,'Five binary outputs and three potency outputs; IKs is exploratory',ha='center',fontsize=7)
    box(15,290,95,60,'chem','SMILES',8)
    box(180,105,300,85,'chem','Three fingerprints: 6,144\nLinear 6,144 → 512 → 256\nBatchNorm · ReLU · dropout')
    box(180,230,300,85,'chem','Chemical descriptors: 20\nTRAIN scaling → signed log\nLinear 20 → 64 → 64')
    arrow(110,305,180,145,-.1);arrow(110,315,180,270)
    ax.add_patch(FancyBboxPatch((145,355),370,220,boxstyle='round,pad=0,rounding_size=5',
        facecolor='#fbf9ff',edgecolor='#a793ca',linestyle='--',lw=1.2))
    ax.text(330,376,'FROZEN PRETRAINED ChemBERTa-MTR',ha='center',fontsize=7,color='#654092')
    box(175,397,310,55,'cb','Final-layer <s> vector: 384',7.5)
    box(175,492,310,60,'cb','Pretrained property head\n384 → 384 → 199 predictions',7.5)
    arrow(110,335,145,425,.12);arrow(330,452,330,492)
    box(565,408,190,145,'cb','Join 384 + 199 = 583\nTRAIN-scale 199\nLinear 583 → 128\nLayerNorm · ReLU\nDropout',7)
    arrow(485,423,565,441);arrow(485,521,565,512)
    box(795,112,135,440,'join','Concatenate\n\n256\n+\n64\n+\n128\n\n448 features',7.5)
    arrow(480,145,795,145);arrow(480,270,795,270);arrow(755,480,795,480)
    outputs=[('hERG 10 µM','cls',120),('hERG 1 µM','cls',178),('hERG pChEMBL','reg',236),
             ('NaV1.5 blocker','cls',304),('NaV1.5 pChEMBL','reg',362),
             ('CaV1.2 blocker','cls',430),('CaV1.2 pChEMBL','reg',488),('IKs blocker*','iks',556)]
    for label,kind,y in outputs:
        box(995,y-21,190,42,kind,label,7.5);arrow(930,330,995,y)
    ax.text(1090,83,'Eight separate heads',ha='center',fontsize=7.5,weight='bold')
    ax.text(600,621,'Each head: Linear 448 → 128 · ReLU · dropout 0.3 · Linear 128 → 1',ha='center',fontsize=7.5)
    ax.text(600,650,'Binary outputs use sigmoid; potency outputs are converted back to pChEMBL.',ha='center',fontsize=7)
    ax.text(600,679,'6,747 input features · 3,820,680 trainable parameters · *IKs reported separately',ha='center',fontsize=7)
    fig.subplots_adjust(left=.015,right=.985,top=.985,bottom=.015)
    out=Path('reports/paper_revision/figures/figure1_architecture')
    fig.savefig(out.with_suffix('.svg'),facecolor='white')
    fig.savefig(out.with_suffix('.png'),dpi=600,facecolor='white')
    fig.savefig(out.with_suffix('.pdf'),facecolor='white')


if __name__=='__main__':main()
