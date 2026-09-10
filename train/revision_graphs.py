"""Compact molecular graph cache with explicit atom/bond feature definitions."""
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem

CHOICES=[list(range(1,101)),list(range(6)),[-2,-1,0,1,2],list(range(4)),list(range(5)),
    [int(t) for t in [Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3,Chem.HybridizationType.SP3D,Chem.HybridizationType.SP3D2]]]
SIZES=[len(c)+1 for c in CHOICES]


def compact_graph(smiles):
    m=Chem.MolFromSmiles(smiles);assert m is not None and m.GetNumAtoms()>0
    atoms=[];mass=[];edges=[];bonds=[]
    for a in m.GetAtoms():
        values=[a.GetAtomicNum(),a.GetTotalDegree(),a.GetFormalCharge(),int(a.GetChiralTag()),a.GetTotalNumHs(),int(a.GetHybridization())]
        atoms.append([c.index(v) if v in c else len(c) for c,v in zip(CHOICES,values)]+[int(a.GetIsAromatic())]);mass.append(a.GetMass()*.01)
    for b in m.GetBonds():
        order=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        feature=[0]+[int(b.GetBondType()==t) for t in order]+[int(b.GetIsConjugated()),int(b.IsInRing())]
        stereo=int(b.GetStereo());feature+=[int(i==(stereo if stereo in range(6) else 6)) for i in range(7)]
        edges.extend([(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),(b.GetEndAtomIdx(),b.GetBeginAtomIdx())]);bonds.extend([feature,feature])
    return np.asarray(atoms,dtype=np.uint8),np.asarray(mass,dtype=np.float32),np.asarray(edges,dtype=np.uint32).reshape(-1,2),np.asarray(bonds,dtype=np.uint8).reshape(-1,14)


class GraphCache:
    def __init__(self,path=Path('data/features/revision_graphs')):
        self.manifest=json.loads((path/'manifest.json').read_text());assert self.manifest['status']=='complete'
        for name in ['atoms','mass','edges','bonds','atom_offsets','edge_offsets']:setattr(self,name,np.load(path/(name+'.npy'),mmap_mode='r'))
        self.lengths=np.diff(self.atom_offsets)

    def batch(self,rows,device):
        atoms=[];mass=[];edges=[];bonds=[];membership=[];node_at=0
        for i,row in enumerate(rows):
            sl=slice(self.atom_offsets[row],self.atom_offsets[row+1]);el=slice(self.edge_offsets[row],self.edge_offsets[row+1])
            aa=self.atoms[sl];atoms.append(aa);mass.append(self.mass[sl]);edges.append(self.edges[el]+node_at);bonds.append(self.bonds[el])
            membership.extend([i]*len(aa));node_at+=len(aa)
        a=np.concatenate(atoms);features=np.zeros((len(a),133),dtype=np.float32);at=0;ind=np.arange(len(a))
        for c,size in enumerate(SIZES):features[ind,at+a[:,c]]=1;at+=size
        assert at==131;features[:,131]=a[:,6];features[:,132]=np.concatenate(mass)
        e=np.concatenate(edges).astype(np.int64);bond=np.concatenate(bonds).astype(np.float32)
        to=lambda x:torch.from_numpy(np.asarray(x)).to(device)
        return (to(features),to(bond),to(e[:,0]),to(e[:,1]),to(np.arange(len(e),dtype=np.int64)^1),
            to(np.asarray(membership,dtype=np.int64)),len(rows))
