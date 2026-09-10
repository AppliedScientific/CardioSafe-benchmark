"""Package existing L1000 evidence only; no new encoder or branch experiments."""
import csv
import json
import re
import shutil
from pathlib import Path
from train.readout_contract import sha


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    root=Path('reports/paper_revision');out=root/'historical_l1000';out.mkdir(exist_ok=True)
    asi=Path('../asi/data/lukas-paper');files=[asi/'asi287_item_7_l1000_graph_density.json',asi/'asi287_items_3_8.json',
        Path('data/supplementary/table_s8_l1000_threshold_sweep.csv'),Path('data/supplementary/table_s8_l1000_threshold_sweep.json')]
    manifest=[]
    for p in files:
        dest=out/p.name;shutil.copyfile(p,dest);manifest.append(dict(source=str(p.resolve()),copy=str(dest),sha256=sha(p)))
    # This excerpt was obtained with pdftotext -f18 -l19 -layout from the supplied PDF.
    excerpt=out/'submitted_ablation_excerpt.txt'
    if not excerpt.exists():shutil.copyfile('/tmp/cardiosafe_revision_ablation_text.txt',excerpt)
    manifest.append(dict(source='Submitted PDF Table4/4b, pdftotext pages18–19',copy=str(excerpt),sha256=sha(excerpt)))
    rows=[];split='tan70'
    pattern=r'^\s*(hERG 10 µM|hERG 1 µM|Nav1.5|Cav1.2|IKs)\s+(\d+)\s+(.+)$'
    for line in excerpt.read_text().splitlines():
        if 'Table 4b' in line:split='tan60'
        match=re.match(pattern,line)
        if not match:continue
        head,n,values=match.groups();triples=re.findall(r'(\d+\.\d+)\s+\[(\d+\.\d+),\s*(\d+\.\d+)\]',values)
        assert len(triples)==3
        for model,(point,lo,hi) in zip(['historical_three_branch','historical_bio_zero','historical_chemberta_zero'],triples):
            rows.append(dict(model=model,head=head,split=split,metric='auc_roc',estimate=float(point),
                ci_lower=float(lo),ci_upper=float(hi),n=int(n),status='Transcribed rounded submitted Table4/4b; historical labels/training; not v3 results'))
    assert len(rows)==27
    write(out/'submitted_branch_ablation.csv',rows)
    graph=json.loads(files[0].read_text());total=graph['total_undirected_pairs'];edges=graph['saved_encoder_undirected_edges_at_default_thr_0_40']
    record=dict(n_genes=graph['n_genes'],predicted_signature_sample_n=graph['subsample_n'],
        graph_pairs=total,reference_edges=edges,reference_density=edges/total,
        predicted_edges=graph['thresholds']['thr_0.40']['n_edges_undirected'],
        predicted_density=graph['thresholds']['thr_0.40']['density'],
        density_ratio=graph['thresholds']['thr_0.40']['n_edges_undirected']/edges,
        interpretation='Different graph construction inputs; descriptive mismatch, not a causal encoder-accuracy experiment')
    (out/'graph_density_summary.json').write_text(json.dumps(record,indent=2))
    attn=json.loads(files[1].read_text());records=[]
    for split,block in attn.items():
        for seed in block['item_8_attn_introspection']['per_seed']:
            for channel,weights in seed['mean_attn_weights'].items():
                records.append(dict(split=split,seed=seed['seed'],channel=channel,**weights,
                    entropy_nats=seed['mean_attn_entropy_nats'][channel],initialization_comparator_available=False))
    write(out/'historical_attention_summary.csv',records)
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
