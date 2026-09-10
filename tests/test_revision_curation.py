import csv
import json
import pytest
from pathlib import Path
from scripts.revision_curation import classify


def test_censored_and_percent_votes_do_not_impute_unmeasured_thresholds():
    assert classify([], [('>',4)], [],5)==0
    assert classify([], [('<',6)], [],5)==1
    assert classify([], [('>',6)], [],5) is None
    assert classify([], [],[(1,20)],5) is None
    assert classify([], [],[(1,20)],6)==0
    assert classify([], [],[(1,80)],5)==1
    assert classify([], [],[(30,80)],5) is None


def test_versioned_corrections_preserve_identities_and_untouched_channels():
    required = [Path("data/labels") / name for name in
                ("labels_v2.csv", "labels_v3.csv", "MANIFEST_v3.json")]
    if not all(path.is_file() for path in required):
        pytest.skip("Curation replay requires separately supplied v2/v3 labels and v3 manifest")
    with open('data/labels/labels_v2.csv') as f:old=list(csv.DictReader(f))
    with open('data/labels/labels_v3.csv') as f:new=list(csv.DictReader(f))
    manifest=json.loads(Path('data/labels/MANIFEST_v3.json').read_text())
    differences=0
    for a,b in zip(old,new,strict=True):
        for key in ['row_idx','inchikey','nav15_pchembl','nav15_blocker','cav12_pchembl','cav12_blocker']:assert a[key]==b[key]
        for key in a:
            if key in ('row_idx','inchikey'):continue
            x=float(a[key]) if a[key] else None;y=float(b[key]) if b[key] else None
            differences+=x!=y
    assert differences==manifest['changed_cells']==262
    # Literal activator-only potency cannot survive via a duplicate curated fallback.
    bykey={r['inchikey']:r for r in new}
    assert bykey['VWNMWKSURFWKAL-HXOBKFHXSA-N']['herg_pchembl']==''
    assert bykey['IHUPABWARJLUMA-UHFFFAOYSA-N']['herg_pchembl']==''
