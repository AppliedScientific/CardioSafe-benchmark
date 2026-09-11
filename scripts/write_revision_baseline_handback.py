"""Produce readable per-head tables from the complete, audited baseline matrix."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.draw_revision_baseline_results import NAMES, HEADS


def interval(row):
    if not np.isfinite(row.estimate):
        return 'undefined'
    return f'{row.estimate:.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]'


def table(lines, data, models, heads, metric, comparison):
    lines += ['| Model | ' + ' | '.join(heads.values()) + ' |',
              '|---|' + '---|' * len(heads)]
    for model in models:
        cells = []
        for head in heads:
            rows = data[(data.model == model) & (data['head'] == head)
                        & (data.metric == metric) & (data.comparison == comparison)]
            assert len(rows) == 1, (model, head, metric, comparison, len(rows))
            cells.append(interval(rows.iloc[0]))
        lines.append('| ' + NAMES.get(model, model) + ' | ' + ' | '.join(cells) + ' |')
    lines.append('')


def main():
    root = Path('reports/paper_revision')
    protocol = json.loads((root / 'tables/phase3_baseline_protocol.json').read_text())
    audit = json.loads((root / 'baseline_inference_audit_all.json').read_text())
    assert protocol['complete'] and audit['status'] == 'passed' and not audit['missing']
    lines = ['# Completed reviewer baseline matrix', '',
             'Five seeds (42–46), unchanged tan70/tan60 splits, corrected v3 labels, '
             'and probability averaging for classification. Entries are ensemble estimates '
             'with 95% molecule-bootstrap intervals (B=1000, seed 20260428). These intervals '
             'condition on the fitted ensembles; they do not include training-seed uncertainty. '
             'Comparisons are unadjusted for multiplicity.', '',
             'The locked final architecture remains chemical branches + final MTR `<s>` + '
             '199 pretrained predictions, with concatenation and eight outputs. This matrix '
             'evaluates that author-selected architecture; it does not retrospectively select '
             'a winner on the inspected test sets.', '',
             'The randomized-control issue remains unresolved. Proceeding with these baselines '
             'was explicitly authorized; it does not establish that class imbalance explains '
             'the residual real-label AUROC. See `phase1_current_pipeline_review.md`.', '',
             'IKs has one TEST positive in each split. Its estimates and potentially degenerate '
             'conditional intervals cannot establish generalization.', '']
    regression = {'herg_pchembl': 'hERG', 'nav15_pchembl': 'NaV1.5', 'cav12_pchembl': 'CaV1.2'}
    for split in ['tan70_v1_1', 'tan60_v1_1']:
        data = pd.read_csv(root / f'tables/phase3_baselines_{split}.csv')
        lines += [f'## {split}', '', '### Classification AUROC', '']
        table(lines, data, NAMES, HEADS, 'auc_roc', 'absolute')
        lines += ['### Classification AUPRC (average precision)', '',
                  'No-skill prevalence: ' + '; '.join(
                      f"{title} {data[(data.model == 'final') & (data['head'] == head) & (data.metric == 'auprc') & (data.comparison == 'absolute')].iloc[0].prevalence:.4f}"
                      for head, title in HEADS.items()) + '.', '']
        table(lines, data, NAMES, HEADS, 'auprc', 'absolute')
        lines += ['### Regression RMSE (pChEMBL)', '']
        table(lines, data, NAMES, regression, 'rmse', 'absolute')
        lines += ['### Paired AUROC differences: baseline minus final', '',
                  'Positive differences favor the baseline. Assess the paired difference '
                  'interval directly; overlap between two separate absolute intervals is '
                  'not the paired comparison.', '']
        table(lines, data, list(NAMES)[1:], HEADS, 'auc_roc', 'delta_vs_final')
        lines += ['### Paired RMSE differences: baseline minus final', '',
                  'Negative differences favor the baseline.', '']
        table(lines, data, list(NAMES)[1:], regression, 'rmse', 'delta_vs_final')
        matched=pd.read_csv(root / f'tables/phase3_adaptation_matched_{split}.csv')
        lines += ['### Fine-tuning versus its matched frozen control', '',
                  'Direct paired differences are fine-tuned minus frozen with matched batching. '
                  'Positive AUROC and negative RMSE differences favor fine-tuning.', '']
        table(lines, matched, ['finetune'], HEADS, 'auc_roc', 'delta_vs_frozen_matched')
        table(lines, matched, ['finetune'], regression, 'rmse', 'delta_vs_frozen_matched')
    lines += ['## Supporting files and limits', '',
              '- Full per-head MCC, sensitivity, specificity, accuracy, MAE, Pearson r, '
              'Spearman rho and wRMSE are in `tables/phase3_baselines_*.csv`.',
              '- Individual seed metrics are in `tables/phase3_baseline_per_seed_all.csv`.',
              '- IC50-only comparisons use the same truth and molecules within each named '
              'cohort; see `tables/phase3_ic50_sensitivity_*.csv` and `sparse_baseline_findings.md`.',
              '- The hERG-volume arms and their paired differences are included in the full '
              'tables and `phase3_transfer_sweep_*` figures.',
              '- Single-channel optimization exposure differs from multi-task training; '
              'see `baseline_methods_and_limits.md` before making a transfer claim.',
              '- Numerical tables are generated from audited exports. Scientific interpretation '
              'and figure layout review remain separate finalization steps.', '']
    (root / 'notes/baseline_per_head_handback.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
