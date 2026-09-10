"""Exact membership of all three ingested sources, restricted to released compounds."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from scripts.revision_curation import read,COLS


def save(path,rows):
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ch=read('chembl36_eligible_evidence.csv');sc=read('hergcentral_records.csv');ct=read('ctoxpred_records.csv')
    sets=dict(ChEMBL36={r['standardized_inchikey'] for r in ch},hERGCentral={r['original_pipeline_inchikey'] for r in sc},CToxPred_curated={r['standardized_inchikey'] for r in ct})
    with open('data/compounds/compounds.csv') as f:compounds=list(csv.DictReader(f))
    membership=[];groups=Counter()
    for r in compounds:
        present=tuple(s for s,keys in sets.items() if r['inchikey'] in keys);assert present,r
        group='+'.join(present);groups[group]+=1
        membership.append(dict(row_idx=r['row_idx'],inchikey=r['inchikey'],source_membership=group,**{s:int(s in present) for s in sets}))
    save('reports/paper_revision/tables/phase2_source_membership.csv',membership)
    save('reports/paper_revision/tables/phase2_source_venn_three_sources.csv',[dict(source_membership=k,unique_released_compounds=v) for k,v in sorted(groups.items())])
    composition=[]
    for version in ['v2','v3']:
        with open(f'data/labels/labels_{version}.csv') as f:labels=list(csv.DictReader(f))
        assert [r['row_idx'] for r in labels]==[r['row_idx'] for r in compounds]
        for group in sorted(groups):
            selected=[r for r,m in zip(labels,membership) if m['source_membership']==group]
            for col in COLS:
                observed=[float(r[col]) for r in selected if r[col]]
                composition.append(dict(label_version=version,source_membership=group,head=col,n_compounds=len(selected),
                    n_labelled=len(observed),n_positive=sum(v==1 for v in observed) if 'blocker' in col else '',
                    n_negative=sum(v==0 for v in observed) if 'blocker' in col else ''))
    save('reports/paper_revision/tables/phase2_source_label_composition.csv',composition)
    pool={r['inchikey'] for r in compounds};stats=[]
    for name,rows,key,channel in [('ChEMBL36',ch,'standardized_inchikey','channel'),('hERGCentral',sc,'original_pipeline_inchikey',None),('CToxPred_curated',ct,'standardized_inchikey','channel')]:
        matched=[r for r in rows if r[key] in pool];pairs={(r[key],r[channel] if channel else 'herg') for r in rows}
        stats.append(dict(source=name,input_rows=len(rows),unique_standardized_compounds=len(sets[name]),
            unique_released_compounds=len(sets[name]&pool),matched_source_rows=len(matched),compound_channel_groups=len(pairs),
            rows_beyond_one_per_compound_channel=len(rows)-len(pairs),
            row_semantics='Individual ChEMBL activities' if name=='ChEMBL36' else 'Paired-concentration screen records' if name=='hERGCentral' else 'Curated compound summaries, with overlap across source files'))
    save('reports/paper_revision/tables/phase2_source_counts.csv',stats)
    report=dict(unique_released_compounds=len(pool),membership_categories=dict(groups),unassigned_source_membership=0,
        two_source_venn_incomplete=True,ctoxpred_only_released=groups['CToxPred_curated'],
        hergcentral_released=len(sets['hERGCentral']&pool),
        caution='Membership is source occurrence, not proof that every source supplied the final value. ChEMBL assay census excludes CToxPred compound summaries whose individual assay metadata are unavailable. Repeated source rows are not necessarily duplicate measurements.')
    Path('reports/paper_revision/notes/phase2_source_provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
