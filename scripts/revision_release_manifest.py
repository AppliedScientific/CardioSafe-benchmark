"""Manifest of the final eight-output weights and verified feature inference."""
from pathlib import Path
import json
import torch
from train.readout_contract import sha


def main():
    root=Path('reports/paper_revision');contract=json.loads((root/'architecture_lock.json').read_text());records=[]
    assert json.loads((root/'release_inference_audit.json').read_text())['status']=='passed'
    for split in ['tan70_v1_1','tan60_v1_1']:
        preprocessing=Path('data/features')/('revision_'+split)/'manifest.json'
        for seed in range(42,47):
            path=Path('runs_revision_v3/final')/split/f'seed_{seed}/model.pt';c=torch.load(path,weights_only=False,map_location='cpu')
            assert c['seed']==seed and c['split']==split and c['labels']=='labels_v3.csv' and c['attention']=='none' and c['y_randomize'] is None
            provenance=path.parent/'provenance.json'
            records.append(dict(split=split,seed=seed,path=str(path),sha256=sha(path),selected_epoch=c['epoch'],validation_wrmse=-c['val_es'],property_preprocessing_path=str(preprocessing),property_preprocessing_sha256=sha(preprocessing),training_provenance_path=str(provenance),training_provenance_sha256=sha(provenance),training_source_sha256=json.loads(provenance.read_text())['source_sha256']))
    files=['model/revision.py','model/latent_fusion.py','model/cardiosafe.py','inference/revision_predictor.py','train/fusion_preprocessing.py','scripts/revision_features.py','scripts/build_mtr_predictions.py']
    value=dict(status='Local final weights and raw-feature inference verified; external services unchanged',architecture_contract_sha256=sha(root/'architecture_lock.json'),
        architecture={k:contract[k] for k in ['input_dim','chemical_dim','first_token_dim','mtr_predictions_dim','trainable_parameters','attention','layer2','l1000','heads']},
        labels_path=contract['labels_file'],labels_sha256=sha(contract['labels_file']),checkpoints=records,
        inference_source_sha256={p:sha(p) for p in files},ensemble='Mean sigmoid scores; mean inverse-scaled pChEMBL',
        frozen_checkpoint=dict(model='DeepChem/ChemBERTa-77M-MTR',revision='66b895cab8adebea0cb59a8effa66b2020f204ca',weights_sha256='7882205587d3985c07e425632d0e57ba0fc2fc778e3b47aab02aef9b7961407f'),
        primary_split='tan70_v1_1',secondary_split='tan60_v1_1',exploratory_output='iks_blocker',stage2=False,
        limitations=['No clinical risk or safety claim','IKs utility not established','Scores not calibrated','Raw-SMILES serving integration remains separate from the verified feature-array adapter','Baseline and independent-control handback completion tracked separately'])
    (root/'final_v3_release_manifest.json').write_text(json.dumps(value,indent=2));print('Manifest written for',len(records),'final checkpoints')


if __name__=='__main__':main()
