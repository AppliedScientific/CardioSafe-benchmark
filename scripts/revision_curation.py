"""Reconstruct source precedence and label decisions from retained individual records.

Historical replay is separate from proposed curation changes. This module never
overwrites the released v2 labels or modifies any split.
"""
import csv
import json
import math
from collections import defaultdict,Counter
from pathlib import Path
from statistics import median

ROOT=Path('data/curation/revision')
COLS=('herg_pchembl','herg_blocker_10um','herg_blocker_1um','nav15_pchembl','nav15_blocker','cav12_pchembl','cav12_blocker','iks_blocker')
CHANNELS={'herg':(0,1,2),'nav15':(3,4),'cav12':(5,6),'iks':(7,)}


def read(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))


def classify(exact,censored,percent,threshold):
    yes=sum(x>threshold for x in exact);no=sum(x<=threshold for x in exact)
    for rel,value in censored:
        if rel in ('>','>=') and value<=threshold:no+=1
        elif rel in ('<','<=') and value>=threshold:yes+=1
    for conc,value in percent:
        if threshold==6:
            if conc==1:
                yes+=value>50;no+=value<=50
        elif conc<=3:
            yes+=value>50
        elif conc==10:
            yes+=value>50;no+=value<=50
        elif conc>=30:no+=value<=50
    return int(yes>no) if yes+no else None


def reconstruct(excluded_assays=(),ic50_only=False,retain_screen=True,ctoxpred=True,screen_policy='v2_identity',quarantine_fallback=True):
    evidence=read('chembl36_eligible_evidence.csv');external=read('ctoxpred_records.csv');screen=read('hergcentral_records.csv')
    profiles={};groups=defaultdict(lambda:dict(exact=[],censored=[],percent=[]));affected=set();counts=Counter()
    for r in evidence:
        pair=(r['standardized_inchikey'],r['channel']);assay=int(r['assay_id'])
        if assay in excluded_assays:
            affected.add(pair);counts['excluded_assay_records']+=1;continue
        if ic50_only and r['standard_type']!='IC50':continue
        e=groups[pair]
        if r['evidence_type']=='potency':
            if r['pchembl_value']:e['exact'].append(float(r['pchembl_value']))
            elif r['standard_value'] and float(r['standard_value'])>0:e['censored'].append((r['standard_relation'],9-math.log10(float(r['standard_value']))))
        elif r['parsed_concentration_um']:e['percent'].append((float(r['parsed_concentration_um']),float(r['standard_value'])))
    for (key,ch),e in groups.items():
        row=profiles.setdefault(key,[None]*8)
        if ch!='iks':row[CHANNELS[ch][0]]=median(e['exact']) if e['exact'] else None
        for col in CHANNELS[ch]:
            if 'blocker' in COLS[col]:row[col]=classify(**e,threshold=6 if col==2 else 5)
    if ctoxpred:
        ex=defaultdict(list)
        for r in external:ex[r['standardized_inchikey'],r['channel']].append(float(r['pIC50']))
        for pair,values in ex.items():
            key,ch=pair
            if quarantine_fallback and pair in affected:
                counts['quarantined_curated_fallback_compound_channels']+=1;continue
            row=profiles.setdefault(key,[None]*8);value=median(values)
            for col in CHANNELS[ch]:
                if row[col] is None:row[col]=value if 'pchembl' in COLS[col] else int(value>(6 if col==2 else 5))
    if retain_screen:
        # Historical DB used first source row, erroneous positive current sign,
        # and COALESCE. Reproduce it before applying the later v2 patch.
        first={}
        for r in screen:first.setdefault(r['original_pipeline_inchikey'],r)
        for key,r in first.items():
            row=profiles.setdefault(key,[None]*8)
            for c,field in [(1,'percent_change_10um'),(2,'percent_change_1um')]:
                if row[c] is None:row[c]=int(float(r[field])>50)
        if screen_policy!='historical_db':
            chosen={};keyfield='v2_pipeline_inchikey' if screen_policy=='v2_identity' else 'original_pipeline_inchikey'
            for r in screen:
                key=r[keyfield]
                if key not in chosen or float(r['percent_change_10um'])<float(chosen[key]['percent_change_10um']):chosen[key]=r
            for key,r in chosen.items():
                row=profiles.setdefault(key,[None]*8);a,b=float(r['percent_change_1um']),float(r['percent_change_10um'])
                votes=[int(b< -50)]
                if a< -50:votes.append(1)
                if row[0] is not None:votes.append(int(round(row[0],4)>5))
                row[1]=int(2*sum(votes)>len(votes));row[2]=int(a< -50)
    return profiles,dict(counts)


def main():
    profiles,counts=reconstruct(quarantine_fallback=False)
    with open('data/labels/labels_v2.csv') as f:released=list(csv.DictReader(f))
    discrepancies=[]
    for r in released:
        values=profiles.get(r['inchikey'],[None]*8)
        for col,name in enumerate(COLS):
            actual=float(r[name]) if r[name] else None;expected=values[col]
            if expected is not None and 'pchembl' in name:expected=round(expected,4)
            if actual!=expected:
                discrepancies.append(dict(row_idx=r['row_idx'],inchikey=r['inchikey'],head=name,released=actual,reconstructed=expected))
    out=Path('reports/paper_revision/tables/phase2_source_replay_discrepancies.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['row_idx','inchikey','head','released','reconstructed']);w.writeheader();w.writerows(discrepancies)
    result=dict(compounds=len(released),mismatching_cells=len(discrepancies),mismatches_by_head=dict(Counter(r['head'] for r in discrepancies)),
        reconstruction='ChEMBL36 exact/censored/inhibition evidence; CToxPred median pIC50 fallback; historical screen COALESCE then v2 label-rebuild rule',
        labels_unchanged=True)
    Path('reports/paper_revision/notes/phase2_source_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
