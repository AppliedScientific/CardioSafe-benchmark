"""Compare compatibility replay with the author's sklearn1.3 pipeline.

Run --prepare in the project environment, then --verify in an isolated old
sklearn runtime. The fixture includes missing descriptors and real molecules.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'assets/revision_comparators/CToxPred2/CToxPred2'
spec=importlib.util.spec_from_file_location('author_correlation',BASE/'pairwise_correlation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CorrelationThreshold=module.CorrelationThreshold
FIXTURE=ROOT/'reports/paper_revision/inputs/ctox_descriptor_fixture.npz'


def pipeline(channel):
    directory,stem={'nav15':('Nav1.5','nav'),'cav12':('Cav1.2','cav')}[channel]
    return joblib.load(BASE/'models/decriptors_preprocessing'/directory/(stem+'_descriptors_preprocessing_pipeline.sav'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:
        sys.path.insert(0,'/tmp/cardiosafe_revision_dependencies')
        from rdkit import Chem
        from mordred import Calculator,descriptors
        from model.revision_comparator_adapters import fixed_descriptor_transform
        smiles=['CCO','c1ccccc1','CC(=O)Oc1ccccc1C(=O)O','CN(C)CCOC(c1ccccc1)c1ccccc1','[Na+]','C[N+](C)(C)CC(=O)[O-]']
        frame=Calculator(descriptors,ignore_3D=True).pandas([Chem.MolFromSmiles(s) for s in smiles],nproc=1,quiet=True).fill_missing(np.nan)
        np.savez_compressed(FIXTURE,data=frame.to_numpy(dtype=float),columns=np.asarray(frame.columns,dtype=str),
            **{c:fixed_descriptor_transform(pipeline(c),frame) for c in ['nav15','cav12']})
        print('prepared',frame.shape);return
    import sklearn
    assert sklearn.__version__=='1.3.1',sklearn.__version__
    f=np.load(FIXTURE);frame=pd.DataFrame(f['data'],columns=f['columns']);results={}
    for c in ['nav15','cav12']:
        expected=pipeline(c).transform(frame)
        np.testing.assert_allclose(f[c],expected,rtol=1e-12,atol=1e-12)
        results[c]=dict(shape=list(expected.shape),maximum_absolute_difference=float(np.max(np.abs(f[c]-expected))))
    result=dict(status='passed',reference_sklearn=sklearn.__version__,numpy=np.__version__,results=results,
        method='Original saved pipeline.transform versus explicit-array compatibility replay; six real molecules including missing descriptors')
    (ROOT/'reports/paper_revision/notes/phase4_ctox_compatibility.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
