"""Compact, row-aligned original SMILES token IDs for supervised encoder comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from train.readout_contract import sha


def main():
    out=Path('data/features/revision_smiles_ids');out.mkdir(exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR',local_files_only=True)
    smiles=pd.read_csv('data/compounds/compounds.csv').smiles.tolist();ids=[];lengths=[]
    for start in range(0,len(smiles),4096):
        batch=tokenizer(smiles[start:start+4096],truncation=True,max_length=512,padding=False,return_attention_mask=False)['input_ids']
        lengths.extend(map(len,batch));ids.extend(v for row in batch for v in row)
    arrays={'ids':np.asarray(ids,dtype=np.int16),'offsets':np.r_[0,np.cumsum(lengths)].astype(np.int64)}
    for name,array in arrays.items():np.save(out/(name+'.npy'),array)
    manifest=dict(status='complete',rows=len(smiles),tokens=len(ids),max_length=512,pad_id=tokenizer.pad_token_id,
        model='DeepChem/ChemBERTa-77M-MTR',compound_sha256=sha('data/compounds/compounds.csv'),source_sha256=sha(__file__),
        output_sha256={name:sha(out/(name+'.npy')) for name in arrays})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest)


if __name__=='__main__':main()
