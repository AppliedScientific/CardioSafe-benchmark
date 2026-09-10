"""Graph comparator must be invariant to atom numbering and batching."""
import numpy as np
import torch
from model.revision_dmpnn import RevisionDMPNN
from train.revision_graphs import compact_graph,SIZES


def tensors(smiles):
    aa,mass,edge,bonds=compact_graph(smiles);a=np.zeros((len(aa),133),dtype=np.float32);start=0
    for col,size in enumerate(SIZES):a[np.arange(len(a)),start+aa[:,col]]=1;start+=size
    a[:,131]=aa[:,6];a[:,132]=mass
    return [torch.from_numpy(a),torch.from_numpy(bonds.astype(np.float32)),torch.from_numpy(edge[:,0].astype(np.int64)),
        torch.from_numpy(edge[:,1].astype(np.int64)),torch.arange(len(edge))^1,torch.zeros(len(a),dtype=torch.long),1]


def test_atom_numbering_and_disjoint_batch_invariance():
    torch.manual_seed(42);model=RevisionDMPNN().eval();g=tensors('CC(O)c1ccccc1');a,b,src,dst,rev,batch,n=g
    perm=torch.randperm(len(a));inverse=torch.argsort(perm)
    reordered=[a[perm],b,inverse[src],inverse[dst],rev,batch,n]
    with torch.no_grad():
        torch.testing.assert_close(model.encode(g),model.encode(reordered))
        two=[torch.cat([a,a]),torch.cat([b,b]),torch.cat([src,src+len(a)]),torch.cat([dst,dst+len(a)]),
            torch.cat([rev,rev+len(rev)]),torch.cat([batch,batch+1]),2]
        torch.testing.assert_close(model.encode(two),model.encode(g).repeat(2,1))


def test_bondless_molecule_and_encoder_gradient():
    model=RevisionDMPNN().train();g=tensors('[Na+]')
    out=model.forward_graph(g);assert len(out)==8
    loss=sum(v.square().sum() for v in out.values());loss.backward()
    assert torch.isfinite(loss) and model.atom_output.weight.grad.norm()>0
    model.zero_grad();g=tensors('CCCC')
    sum(v.square().sum() for v in model.forward_graph(g).values()).backward()
    assert model.edge_input.weight.grad.norm()>0 and model.edge_update.weight.grad.norm()>0
