"""Worked hERG examples: shared hashed-bit environments and full endpoint scores."""
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.draw_revision_performance import save
from train.readout_contract import sha


def main():
    root=Path('reports/paper_revision');paths=[root/'tables'/s for s in ['phase4_drug_panel_identity.csv','phase4_drug_morgan_environments_both.csv','phase4_drug_descriptor_attributions_both.csv','phase4_drug_predictions_both.csv']]
    panel,maps,descs,preds=[pd.read_csv(p) for p in paths];head='herg_blocker_10um'
    out=root/'figures/drug_examples';out.mkdir(exist_ok=True);records=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    for split in ['tan70_v1_1','tan60_v1_1']:
        for r in panel[panel.split==split].to_dict('records'):
            name=r['compound'];m=Chem.MolFromSmiles(r['smiles']);fig,axes=plt.subplots(2,2,figsize=(7.1,5.6),layout='constrained')
            local=maps[(maps.compound==name)&(maps.split==split)&(maps['head']==head)];bits=local.drop_duplicates('bit').sort_values('bit_attribution')
            for column,positive in enumerate([True,False]):
                ax=axes[0,column];candidates=bits[bits.bit_attribution>0] if positive else bits[bits.bit_attribution<0]
                selected=candidates.tail(1) if positive else candidates.head(1)
                if selected.empty:ax.axis('off');ax.text(.1,.5,'No selected bit of this sign');continue
                bit=int(selected.bit.iloc[0]);value=float(selected.bit_attribution.iloc[0]);env=local[local.bit==bit]
                atoms=sorted({int(i) for text in env.atom_indices for i in text.split(';')})
                bonds=sorted({int(b) for rr in env.itertuples() for b in Chem.FindAtomEnvironmentOfRadiusN(m,int(rr.radius),int(rr.center))})
                color=(1.,.75,.65) if positive else (.68,.80,1.);stem=f'{split}_{name}_'+('positive' if positive else 'negative')
                kwargs=dict(highlightAtoms=atoms,highlightBonds=bonds,highlightAtomColors={i:color for i in atoms},highlightBondColors={i:color for i in bonds})
                svg=rdMolDraw2D.MolDraw2DSVG(720,350);svg.drawOptions().clearBackground=False;svg.drawOptions().fixedFontSize=21;rdMolDraw2D.PrepareAndDrawMolecule(svg,m,**kwargs);svg.FinishDrawing();(out/(stem+'.svg')).write_text(svg.GetDrawingText())
                png=rdMolDraw2D.MolDraw2DCairo(2160,1050);png.drawOptions().bondLineWidth=4.5;png.drawOptions().fixedFontSize=64;rdMolDraw2D.PrepareAndDrawMolecule(png,m,**kwargs);png.FinishDrawing();data=png.GetDrawingText();(out/(stem+'.png')).write_bytes(data)
                ax.imshow(plt.imread(io.BytesIO(data),format='png'));ax.axis('off');ax.set_title(f'Bit {bit}: IG {value:+.3f}\n{len(env)} shared environment(s)',fontsize=8)
                records.append(dict(compound=name,split=split,fold=r['fold'],head=head,selected_sign='positive' if positive else 'negative',bit=bit,attribution=value,environments=len(env),source_svg=str(out/(stem+'.svg'))))
            ax=axes[1,0];d=descs[(descs.compound==name)&(descs.split==split)&(descs['head']==head)].copy();d['absolute']=d.attribution.abs();d=d.nlargest(5,'absolute').sort_values('attribution')
            ax.barh(d.feature,d.attribution,color=['#ba6348' if v>0 else '#4f76a6' for v in d.attribution]);ax.axvline(0,color='.7',lw=.7);ax.set_xlabel('Descriptor IG (logit units)');ax.set_title('Largest descriptor contributions',fontsize=8)
            ax=axes[1,1];ax.axis('off');p=preds[(preds.compound==name)&(preds.split==split)]
            aliases={'herg_pchembl':'hERG pChEMBL','herg_blocker_10um':'hERG 10 µM score','herg_blocker_1um':'hERG 1 µM score','nav15_pchembl':'NaV1.5 pChEMBL','nav15_blocker':'NaV1.5 score','cav12_pchembl':'CaV1.2 pChEMBL','cav12_blocker':'CaV1.2 score','iks_blocker':'IKs score'}
            cells=[[aliases[q.head],f'{q.prediction:.3f}','—' if pd.isna(q.observed) else f'{q.observed:.3g}'] for q in p.itertuples()]
            table=ax.table(cellText=cells,colLabels=['Endpoint','Model','Label'],cellLoc='left',colWidths=[.59,.21,.20],bbox=[0,.14,1,.84]);table.auto_set_font_size(False);table.set_fontsize(7)
            for (i,j),cell in table.get_celld().items():cell.set_linewidth(.3);cell.set_edgecolor('.8');cell.set_facecolor('#edf1f5' if i==0 else 'white')
            ax.text(0,.02,'Scores are uncalibrated. IKs is exploratory.\nBinary votes and potency medians can disagree.',fontsize=7,transform=ax.transAxes)
            fig.suptitle(f'{name} • {split[:5]} • {r["fold"].upper()} example\nConditional chemical IG; MTR held fixed; all bit environments highlighted',fontsize=9)
            save(fig,root/'figures'/f'phase4_drug_example_{split}_{name}')
    pd.DataFrame(records).to_csv(root/'tables/phase4_drug_figure_selection_both.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(input_sha256={str(p):sha(p) for p in paths},selection='Largest positive and largest negative attribution among the saved top20 absolute Morgan bits for hERG10; all environments of each selected bit highlighted',semantics='Conditional on fixed MTR features; bit-level, not atom-specific causal attribution. Individual molecular SVGs are vector; assembled molecular panels use high-resolution rasters.'),indent=2))


if __name__=='__main__':main()
