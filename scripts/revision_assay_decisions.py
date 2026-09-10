"""Explicit assay-direction decisions for every source-eligible EC50 assay."""
import csv
import json
from collections import defaultdict,Counter
from pathlib import Path

EXCLUDED={
    477489:('activation','Voltage-dependent activation endpoint, not blockade'),
    477490:('activation','Increased peak tail current endpoint, not blockade'),
    688226:('protection_from_block','PubChem AID2121 protocol defines EC50 as increased flux under dofetilide block'),
    688728:('protection_from_block','PubChem AID2638 measures reversal of dofetilide-mediated inhibition'),
    1520988:('activation','Kv7.1/KCNE1 activation; an inactive opener is not evidence of absence of blockade'),
    1540367:('dissociation_kinetics','EC50 for induction of radioligand dissociation is not blockade potency or equilibrium affinity'),
    1869833:('activation','Assay explicitly measures ERG activation'),
    2152987:('reduced_inactivation','Nemertide sodium-channel gating modulation; source article describes reduced inactivation'),
    493154:('unresolved_direction','Generic activity description; original functional endpoint not independently resolved'),
    497893:('unresolved_direction','Generic activity description; original functional endpoint not independently resolved'),
    2306464:('unresolved_direction','Safety47 panel description does not establish direction of the EC50 endpoint'),
}
SOURCES={477489:'https://doi.org/10.1073/pnas.0703934104',477490:'https://doi.org/10.1073/pnas.0703934104',
    688226:'https://pubchem.ncbi.nlm.nih.gov/bioassay/2121',688728:'https://pubchem.ncbi.nlm.nih.gov/bioassay/2638',
    1520988:'https://doi.org/10.1016/j.bmcl.2015.04.074',1540367:'https://doi.org/10.1016/j.ejmech.2015.10.032',
    1869833:'https://doi.org/10.1016/j.bmcl.2019.01.024',2152987:'https://pubs.acs.org/doi/10.1021/acs.jnatprod.1c00104'}


def main():
    with open('data/curation/revision/chembl36_eligible_evidence.csv') as f:rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows:
        if r['standard_type']=='EC50':groups[int(r['assay_id'])].append(r)
    decisions=[]
    for assay,records in groups.items():
        r=records[0];description=r['assay_description'];lower=description.lower()
        if assay in EXCLUDED:category,reason=EXCLUDED[assay];decision='exclude'
        elif any(t in lower for t in ['binding','displacement','selectivity interaction']):
            category='binding';reason='Retained only in mixed-potency policy; not claimed to be functional IC50';decision='retain'
        elif any(t in lower for t in ['inhibition','blockade','antagonist']):
            category='inhibition';reason='Assay description explicitly indicates inhibition/blockade';decision='retain'
        else:raise ValueError(('Unreviewed EC50 assay',assay,description))
        decisions.append(dict(assay_id=assay,assay_chembl_id=r['assay_chembl_id'],channel=r['channel'],
            n_source_records=len(records),n_released_records=sum(x['released_row_idx']!='' for x in records),
            decision=decision,category=category,reason=reason,assay_description=description,
            source_url=SOURCES.get(assay,'https://www.ebi.ac.uk/chembl/explore/assay/'+r['assay_chembl_id'])))
    out=Path('reports/paper_revision/tables/phase2_ec50_assay_decisions.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
    summary=dict(assays=len(decisions),records=sum(r['n_source_records'] for r in decisions),
        excluded_assays=sorted(EXCLUDED),excluded_records=sum(r['n_source_records'] for r in decisions if r['decision']=='exclude'),
        excluded_released_records=sum(r['n_released_records'] for r in decisions if r['decision']=='exclude'),
        excluded_records_by_channel=dict(Counter(r['channel'] for r in rows if int(r['assay_id']) in EXCLUDED)),
        policy='Remove demonstrated non-blockade EC50 assays and unresolved EC50 direction; retain explicit blockade and binding proxies in primary mixed policy. Strict IC50 sensitivity excludes all EC50/Ki/Kd/AC50/XC50 and unauditable curated summaries.')
    Path('reports/paper_revision/notes/phase2_ec50_decisions.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
