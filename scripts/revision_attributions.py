"""Chemical-input integrated gradients conditional on the observed MTR branch.

The path moves fingerprints/descriptors from zero in processed feature space,
holding CLS/MTR inputs fixed. This is an attribution convention, not a chemical
intervention or a physically realizable molecular transformation.
"""
import argparse
import csv
import json
import functools
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from scipy.special import roots_legendre
from train.readout_contract import sha
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scripts.revision_model_io import load_final
from train.train_revision import read_split,read_labels,LABEL_TO_HEAD,CLS_COLS

DESCRIPTORS=[r['column_id'] for r in json.loads(Path('data/supplementary/table_s0_summary.json').read_text())['rows']]


@functools.lru_cache(maxsize=8)
def quadrature(steps):return roots_legendre(steps)


def integrated_gradients(model,x,head,steps):
    baseline=x.clone();baseline[:,:6164]=0;delta=x-baseline
    nodes,weights=quadrature(steps)
    alpha=torch.as_tensor((nodes+1)/2,dtype=x.dtype,device=x.device)
    weight=torch.as_tensor(weights/2,dtype=x.dtype,device=x.device)
    points=(baseline[None]+alpha[:,None,None]*delta[None]).reshape(-1,x.shape[1]).requires_grad_(True)
    gradient=torch.autograd.grad(model(points)[head].sum(),points)[0].reshape(steps,len(x),-1)
    attr=delta*(gradient*weight[:,None,None]).sum(0)
    with torch.no_grad():gap=model(x)[head]-model(baseline)[head]
    residual=attr.sum(1)-gap
    return attr[:,:6164].detach().cpu().numpy(),gap.cpu().numpy(),residual.detach().cpu().numpy()


def write(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='mps');p.add_argument('--repair',action='store_true');a=p.parse_args();torch.set_num_threads(3)
    out=Path('runs_revision_attributions');out.mkdir(exist_ok=True)
    labels=read_labels(Path('data/labels/labels_v3.csv'),334444)
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('row_idx')
    panel=pd.read_csv('reports/paper_revision/tables/phase4_drug_panel_identity.csv')
    fingerprints=np.load('data/features/morgan_fps_2048.npy',mmap_mode='r');gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    global_rows=[];drug_rows=[];mapping_rows=[];enrichment=[];audits=[];repairs=[]
    prior={}
    if a.repair:
        table=pd.read_csv('reports/paper_revision/tables/phase4_ig_completeness_both.csv')
        prior={(r.split,r.head,r.seed):r.max_quadrature_steps for r in table.itertuples()}
    for split in ['tan70_v1_1','tan60_v1_1']:
        folds=read_split(Path('data/splits')/(split+'.csv'));test=folds['test'];panel_part=panel[panel.split.eq(split)]
        for col in CLS_COLS:
            head=LABEL_TO_HEAD[col];rng=np.random.default_rng(20260428);sample=[]
            for value in [0,1]:
                candidates=test[labels[test,col]==value]
                sample.extend(rng.choice(candidates,min(128,len(candidates)),replace=False))
            sample=np.sort(sample);ids=np.r_[sample,panel_part.row_idx.to_numpy()];assert len(np.unique(ids))==len(ids)
            per_seed=[]
            for seed in range(42,47):
                model,provider,ck=load_final(Path('runs_revision_v3/final')/split/f'seed_{seed}',a.device)
                for parameter in model.parameters():parameter.requires_grad_(False)
                pieces=[];gaps=[];residuals=[];used=[]
                path=out/f'{split}_{head}_seed{seed}.npz'
                if a.repair:
                    cached=np.load(path);np.testing.assert_array_equal(cached['idx'],ids)
                    values=cached['attributions'].copy();gaps=cached['logit_gap'].copy();residuals=cached['completeness_residual'].copy()
                    used=[int(prior[split,head,seed])];bad=np.flatnonzero(np.abs(residuals)>.02*(1+np.abs(gaps)))
                    if len(bad):
                        original=out/'initial_quadrature'/path.name;original.parent.mkdir(exist_ok=True)
                        if not original.exists():shutil.copyfile(path,original)
                        before=sha(path)
                        for position in bad:
                            x=torch.from_numpy(provider[ids[position:position+1]]).to(a.device)
                            for steps in [512,1024,2048,4096]:
                                attr,gap,residual=integrated_gradients(model,x,head,steps)
                                if np.all(np.abs(residual)<=.005*(1+np.abs(gap))):break
                            values[position]=attr[0];gaps[position]=gap[0];residuals[position]=residual[0];used.append(steps)
                        repairs.append(dict(file=str(path),previous_sha256=before,repaired_rows=len(bad),maximum_nodes=max(used)))
                else:
                    for start in range(0,len(ids),16):
                        x=torch.from_numpy(provider[ids[start:start+16]]).to(a.device)
                        for steps in [64,128,256]:
                            attr,gap,residual=integrated_gradients(model,x,head,steps)
                            if np.all(np.abs(residual)<=.02*(1+np.abs(gap))):break
                        pieces.append(attr);gaps.extend(gap);residuals.extend(residual);used.append(steps)
                    values=np.concatenate(pieces)
                per_seed.append(values)
                np.savez_compressed(path,idx=ids,attributions=values,logit_gap=np.asarray(gaps),
                    completeness_residual=np.asarray(residuals),test_sample_n=len(sample),quadrature_max=max(used))
                audits.append(dict(split=split,head=head,seed=seed,rows=len(ids),max_abs_completeness_error=float(np.max(np.abs(residuals))),
                    max_quadrature_steps=max(used),failed_tolerance_n=int(np.sum(np.abs(residuals)>.02*(1+np.abs(gaps))))))
            stack=np.stack(per_seed);average=stack.mean(0);global_attr=average[:len(sample)]
            for start,end,block in [(0,2048,'morgan'),(2048,4096,'atompair'),(4096,6144,'toptorsion'),(6144,6164,'descriptors')]:
                global_rows.append(dict(split=split,head=head,feature=block,kind='block',n=len(sample),
                    mean_abs_attribution=float(np.abs(global_attr[:,start:end]).sum(1).mean()),mean_signed_attribution=float(global_attr[:,start:end].sum(1).mean())))
            descriptor_importance=np.abs(global_attr[:,6144:]).mean(0)
            for i,name in enumerate(DESCRIPTORS):
                global_rows.append(dict(split=split,head=head,feature=name,kind='descriptor',n=len(sample),
                    mean_abs_attribution=float(descriptor_importance[i]),mean_signed_attribution=float(global_attr[:,6144+i].mean())))
            top=set(np.argsort(descriptor_importance)[-5:]);canonical={0,1,6,16};hits=len(top&canonical)
            odds,pvalue=fisher_exact([[hits,5-hits],[4-hits,11+hits]],alternative='greater')
            enrichment.append(dict(split=split,head=head,n=len(sample),top5_canonical_hits=hits,
                canonical_features='mw;clogp;aromatic_rings;frac_cation',odds_ratio=odds,unadjusted_p=pvalue,
                caveat='Exploratory enrichment among20 fixed descriptors; no causal or clinical interpretation'))
            for i,r in enumerate(panel_part.to_dict('records'),len(sample)):
                molecule=Chem.MolFromSmiles(r['smiles']);additional=rdFingerprintGenerator.AdditionalOutput();additional.AllocateBitInfoMap()
                fp=gen.GetFingerprint(molecule,additionalOutput=additional);array=np.zeros(2048,dtype=np.uint8);DataStructs.ConvertToNumpyArray(fp,array)
                np.testing.assert_array_equal(array,fingerprints[r['row_idx']]);bits=additional.GetBitInfoMap()
                for j,name in enumerate(DESCRIPTORS):drug_rows.append(dict(compound=r['compound'],split=split,fold=r['fold'],head=head,feature=name,attribution=float(average[i,6144+j])))
                selected=sorted(bits,key=lambda b:abs(average[i,b]),reverse=True)[:20]
                for bit in selected:
                    for center,radius in bits[bit]:
                        bonds=list(Chem.FindAtomEnvironmentOfRadiusN(molecule,radius,center));atoms={center}
                        for bond in bonds:
                            b=molecule.GetBondWithIdx(bond);atoms.update([b.GetBeginAtomIdx(),b.GetEndAtomIdx()])
                        fragment=Chem.MolFragmentToSmiles(molecule,atomsToUse=sorted(atoms),bondsToUse=bonds,canonical=True)
                        mapping_rows.append(dict(compound=r['compound'],split=split,fold=r['fold'],head=head,bit=bit,
                            bit_attribution=float(average[i,bit]),center=center,radius=radius,atom_indices=';'.join(map(str,sorted(atoms))),
                            environment_smiles=fragment,possible_environments_for_bit=len(bits[bit]),
                            attribution_semantics='Shared hashed-bit attribution; not independently assigned to each environment'))
            print(split,head,'attribution complete',len(sample),flush=True)
    root=Path('reports/paper_revision')
    for name,rows in [('phase4_attribution_importance_both',global_rows),('phase4_drug_descriptor_attributions_both',drug_rows),
        ('phase4_drug_morgan_environments_both',mapping_rows),('phase4_pharmacophore_enrichment_both',enrichment),('phase4_ig_completeness_both',audits)]:write(root/'tables'/(name+'.csv'),rows)
    (root/'notes/phase4_attribution_protocol.json').write_text(json.dumps(dict(
        target='Classification logit; mean integrated gradients across five fitted models',
        baseline='All6164 processed chemical features zero; observed CLS384/MTR199 held fixed',
        sample='Up to128 test positives and128 test negatives per head, seed20260428; named panel scored separately',
        quadrature='Gauss–Legendre64–256 nodes; failed cases refined through4096 nodes; completeness checked and reported',
        limitation='Balanced diagnostic sample, not population-weighted feature importance; hashed bits can encode multiple environments; interpolated inputs are not physical molecules'),indent=2))
    if a.repair:(root/'notes/phase4_ig_refinement.json').write_text(json.dumps(dict(repairs=repairs,remaining_failed_rows=sum(r['failed_tolerance_n'] for r in audits)),indent=2))


if __name__=='__main__':main()
