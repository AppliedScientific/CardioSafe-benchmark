"""Recover the exact CardioGenAI vocabulary from its authors' public source CSV."""
import ast
import csv
import json
from pathlib import Path
import re
from train.readout_contract import sha


def main():
    base=Path('assets/revision_comparators/CardioGenAI');source=base/'src/Transformer.py'
    tree=ast.parse(source.read_text());patterns=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='compile' and node.args:
            value=ast.literal_eval(node.args[0])
            if '<MASK>' in value:patterns.append(value)
    assert len(set(patterns))==1;pattern=patterns[0];regex=re.compile(pattern)
    vocab={'<MASK>','<pad>','[CLS]','[EOS]'};n=0;maxlen=0
    path=Path('/tmp/cardiosafe_revision_cga_transformer.csv')
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            tokens=regex.findall(row['SMILES'].strip());vocab.update(tokens);n+=1;maxlen=max(maxlen,len(tokens))
    import torch
    p=base/'model_parameters/transformer_model_parameters/Bidirectional_Transformer_parameters.pt'
    ck=torch.load(p,map_location='cpu',weights_only=True)
    print({k:tuple(v.shape) for k,v in ck['model_state_dict'].items() if 'embed' in k},flush=True)
    record=dict(vocab=sorted(vocab),rows=n,max_tokens=maxlen,regex=pattern,block_size=135,
        source_csv_sha256=sha(path),source_code_sha256=sha(source),checkpoint_sha256=sha(p))
    (base/'vocabulary.json').write_text(json.dumps(record,indent=2));print('vocabulary',len(vocab),'rows',n,flush=True)


if __name__=='__main__':main()
