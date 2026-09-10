"""Pinned official MolCLR GIN loader and label-free graph featurization.

The graph mapping follows the released dataset code. The default matches the
node/subgraph pretraining paths (implicit H); mixed pretraining and downstream
code use AddHs. The released checkpoint does not record its augmentation path.
The upstream implementation and MIT license are preserved alongside the assets.
"""
import importlib.util
import sys
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Data
from train.readout_contract import sha

SOURCE_SHA = '5acb1a02aef6042d71104985e427d2a91813273f2c94c4f2fe3a36da895cd26f'
WEIGHTS_SHA = '93bc4f02ea8847cd44fa21ec3f65600ff2f4a7ae6d3a85e8519a5bcc56afc20a'
CHIRALITY = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
BONDS = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
DIRECTIONS = (Chem.BondDir.NONE, Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT)


def graph(molecule, explicit_h=False):
    """Preserve atom order and stereochemistry; never drop unsupported atoms."""
    if isinstance(molecule, str):
        molecule = Chem.MolFromSmiles(molecule)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty molecule')
    if explicit_h:
        molecule = Chem.AddHs(molecule)
    atoms = []
    for atom in molecule.GetAtoms():
        if not 1 <= atom.GetAtomicNum() <= 118 or atom.GetChiralTag() not in CHIRALITY:
            raise ValueError('Atom type/chirality is outside the pretrained vocabulary')
        atoms.append((atom.GetAtomicNum()-1, CHIRALITY.index(atom.GetChiralTag())))
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in BONDS or bond.GetBondDir() not in DIRECTIONS:
            raise ValueError('Bond type/direction is outside the pretrained vocabulary')
        a,b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((a,b),(b,a)))
        attributes.extend([(BONDS.index(bond.GetBondType()),DIRECTIONS.index(bond.GetBondDir()))]*2)
    return Data(x=torch.tensor(atoms,dtype=torch.long),
                edge_index=torch.tensor(edges,dtype=torch.long).reshape(-1,2).T.contiguous(),
                edge_attr=torch.tensor(attributes,dtype=torch.long).reshape(-1,2))


def load_frozen(root=Path('assets/molclr_3d3bc1')):
    root = Path(root)
    if sha(root/'ginet_molclr.py') != SOURCE_SHA or sha(root/'model.pth') != WEIGHTS_SHA:
        raise ValueError('MolCLR source/checkpoint differs from the pinned official release')
    name = 'cardiosafe_pinned_molclr_3d3bc1'
    spec = importlib.util.spec_from_file_location(name,root/'ginet_molclr.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # PyG inspects the defining module for message signatures.
    spec.loader.exec_module(module)
    model = module.GINet(num_layer=5,emb_dim=300,feat_dim=512,drop_ratio=0,pool='mean')
    model.load_state_dict(torch.load(root/'model.pth',map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False)
