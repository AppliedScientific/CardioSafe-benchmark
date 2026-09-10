"""Architecture-only graphical abstract; no unresolved performance claims."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scripts.draw_revision_performance import save


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.1,3.4));ax.set(xlim=(0,1000),ylim=(420,0));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=5',facecolor=color,edgecolor='#596575',lw=.9));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,linespacing=1.5)
    def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='->',mutation_scale=10,color='#596575',lw=1))
    ax.text(500,25,'CardioSafe: molecular structure to ion-channel predictions',ha='center',fontsize=11,weight='bold')
    box(15,180,100,60,'SMILES','#e8edf2');box(175,90,270,100,'Chemical features\nThree fingerprints: 6,144 bits\n20 molecular descriptors','#e4edff')
    box(175,240,270,110,'Frozen ChemBERTa-MTR\nFinal <s> vector: 384\nPretrained predictions: 199','#efe8ff')
    box(510,145,180,160,'Trainable projections\n256 + 64 + 128\n↓\nConcatenate: 448','#fff0dc')
    box(755,80,225,265,'Eight output heads\n\nhERG: 3 endpoints\nNaV1.5: 2 endpoints\nCaV1.2: 2 endpoints\nIKs: 1 exploratory','#fff3c3')
    arrow(115,195,175,140);arrow(115,220,175,290);arrow(445,140,510,190);arrow(445,290,510,265);arrow(690,225,755,225)
    ax.text(500,392,'Five blockade scores and three pChEMBL potency estimates • In vitro endpoints',ha='center',fontsize=8)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01);save(fig,Path('reports/paper_revision/figures/graphical_abstract'))


if __name__=='__main__':main()
