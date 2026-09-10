"""Save row-level ensemble predictions, paired differences, and route interventions.

Defaults to validation. Test requires an explicit --fold test after selection.
Bootstrap draws entire molecule rows jointly across heads, preserving label overlap.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from model.cardiosafe import CardioSafe
from model.fusion_factory import build_model, ROUTED_MODES, SELF_MODES, BIT_MODES, MOLECULAR_MODES, parameter_digest
from train.features import build_provider
from train.readout_contract import resolve_readout
from train.token_cache import open_token_cache
from train.molecular_tokens import molecular_probe, molecular_predictions
from train.fusion_preprocessing import transform_descriptors
from train.train_cardiosafe import REPO, read_split, read_labels, LABEL_TO_HEAD, REG_COLS, WRMSE_WEIGHT


def metrics(y, p):
    out = {}
    for c, h in LABEL_TO_HEAD.items():
        mask = np.isfinite(y[:, c])
        if c in REG_COLS:
            out[h] = float(np.sqrt(np.mean((y[mask, c] - p[mask, c]) ** 2)))
        elif len(np.unique(y[mask, c])) == 2:
            out[h] = float(roc_auc_score(y[mask, c], p[mask, c]))
    out['wrmse'] = sum(w * out[h] for h, w in WRMSE_WEIGHT.items())
    return out


def paired_bootstrap(y, p, q, n_boot=1000):
    """Difference p-q; negative is better. Shared draws across heads and models."""
    rng = np.random.default_rng(20260428)
    # Rows carrying no regression labels never affect wRMSE. Retain them in the
    # multinomial draw via their aggregate probability for the exact row bootstrap.
    active = np.isfinite(y[:, REG_COLS]).any(1)
    yy, pp, qq = y[active], p[active], q[active]
    probs = np.r_[np.full(active.sum(), 1 / len(y)), (~active).sum() / len(y)]
    probs /= probs.sum()
    draws = rng.multinomial(len(y), probs, size=n_boot)[:, :-1]
    delta = np.zeros(n_boot)
    per_head = {}
    for c, h in LABEL_TO_HEAD.items():
        if c not in REG_COLS:
            continue
        mask = np.isfinite(yy[:, c])
        count = draws[:, mask]
        denom = count.sum(1)
        if np.any(denom == 0):
            raise ValueError('Bootstrap resample has no observations for a required head')
        ep, eq = (pp[mask, c]-yy[mask, c])**2, (qq[mask, c]-yy[mask, c])**2
        head_delta = np.sqrt(count @ ep / denom)-np.sqrt(count @ eq / denom)
        delta += WRMSE_WEIGHT[h] * head_delta
        per_head[h] = {'delta_rmse':float(np.sqrt(ep.mean())-np.sqrt(eq.mean())),
                       'ci95':np.quantile(head_delta,[.025,.975]).tolist(),
                       'label_count':int(mask.sum())}
    return {'delta_wrmse': metrics(y, p)['wrmse']-metrics(y, q)['wrmse'],
            'ci95': np.quantile(delta, [.025, .975]).tolist(), 'bootstrap': n_boot,
            'seed': 20260428, 'per_regression_head':per_head}


@torch.no_grad()
def routing_probe(model, x):
    """Same feature probe at initialization and after training; no fitted val means.

    Prediction sensitivity is descriptive. Only the full-fold, labeled route
    interventions below measure predictive utility.
    """
    if getattr(model, 'mode', None) in BIT_MODES:
        queries = model.queries(x)
        perm = np.random.default_rng(101).permutation(len(x))
        outputs = {}
        for name in ('native', 'uniform', 'shuffle_101'):
            blocks = [model.forward_with_queries(z, queries[perm][s:s+256] if name == 'shuffle_101' else queries[s:s+256],
                                                uniform=name == 'uniform') for s, z in zip(range(0, len(x), 256), x.split(256))]
            outputs[name] = {h: torch.cat([b[h] for b in blocks]) for h in blocks[0]}
        stats = model.attention_stats(x)
        stats['prediction_rms_change_z_or_logit'] = {
            name: {h: float((o[h]-outputs['native'][h]).square().mean().sqrt()) for h in o}
            for name, o in outputs.items() if name != 'native'}
        return stats
    tokens = model.tokens(x)
    weights = model.routes(tokens)
    native = model.from_tokens(tokens, weights)
    perm = np.random.default_rng(101).permutation(len(x))
    changes = {}
    for name, w in [('uniform', torch.full_like(weights, 1/weights.shape[-1])),
                    ('shuffle_101', weights[perm])]:
        out = model.from_tokens(tokens, w)
        changes[name] = {h: float((out[h]-native[h]).square().mean().sqrt()) for h in native}
    return {'per_molecule_sd': float(weights.std(0).mean()),
            'prediction_rms_change_z_or_logit': changes}


@torch.no_grad()
def predict_run(run, fold, interventions):
    if (run.parent.parent/'stopped_for_packing.json').exists():
        raise ValueError('This unfinished diagnostic cohort is excluded from candidate evaluation')
    ck = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
    mode = ck['attention']
    model = build_model(mode=mode, slots=ck.get('slots', 4), chemberta=ck.get('chemberta', True), packed_tokens=ck.get('packed_tokens', False), encoder_dim=ck.get('encoder_dim', 384))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    folds = read_split(REPO / 'data/splits' / (ck['split'] + '.csv'))
    idx = folds[fold]
    d = np.load(REPO / 'data/features/descriptors_20.npy')
    d = (d - ck['descriptor_scaler_means']) / ck['descriptor_scaler_stds']
    d = transform_descriptors(d, ck.get('descriptor_transform', 'none'))
    provider = build_provider(REPO / 'data/features', d,
                              chemberta_path=resolve_readout(ck.get('chemberta_cache'), REPO))
    if provider.encoder_dim != ck.get('encoder_dim', 384):
        raise ValueError('Checkpoint encoder width disagrees with its cache')
    labels = read_labels(REPO / 'data/labels' / ck['labels'], len(provider))
    if ck.get('y_randomize') is not None:
        labels = labels[np.random.default_rng(ck['y_randomize']).permutation(len(labels))]
    y = labels[idx]
    routed = mode in ROUTED_MODES or mode in BIT_MODES
    stats, preds = {}, {}
    probe_comparison = {}
    if routed:
        probe_x = torch.from_numpy(provider[folds['val'][:4096]])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ck['seed'])
            initial = build_model(mode=mode, slots=ck.get('slots', 4), chemberta=ck.get('chemberta', True), packed_tokens=ck.get('packed_tokens', False), encoder_dim=ck.get('encoder_dim', 384)).eval()
        initial_probe = routing_probe(initial, probe_x)
        provenance_path = run/'provenance.json'
        expected_hash = json.loads(provenance_path.read_text()).get('initial_parameter_sha256') if provenance_path.exists() else None
        if expected_hash is not None and parameter_digest(initial) != expected_hash:
            raise ValueError(f'Initial parameter replay mismatch for {run}')
        recorded = json.loads((run/'initial_routing.json').read_text())
        replay_error = abs(initial_probe['per_molecule_sd']-recorded['per_molecule_sd'])
        if replay_error > 1e-5:
            raise ValueError(f'Initialization replay mismatch for {run}: {replay_error}')
        probe_comparison = {'fold': 'val', 'rows': len(probe_x),
                            'initial': initial_probe, 'trained': routing_probe(model, probe_x),
                            'initial_parameter_hash_verified': True if expected_hash is not None else None,
                            'initial_sd_replay_error': replay_error}
    if mode in MOLECULAR_MODES:
        contract = ck.get('token_cache')
        if contract is None: raise ValueError('Molecular checkpoint has no token contract')
        cache, _ = open_token_cache(REPO/contract['path'], REPO, expected=contract)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ck['seed'])
            initial = build_model(mode=mode, packed_tokens=ck.get('packed_tokens', False), encoder_dim=ck.get('encoder_dim', 384)).eval()
        expected_hash = json.loads((run/'provenance.json').read_text())['initial_parameter_sha256']
        if parameter_digest(initial) != expected_hash: raise ValueError('Initial parameter replay mismatch')
        probe_rows = folds['val'][:1024]
        initial_probe = molecular_probe(initial,provider,probe_rows,cache)
        recorded = json.loads((run/'initial_routing.json').read_text())
        for key in ('query_per_molecule_sd','mean_normalized_entropy'):
            if abs(recorded[key]-initial_probe[key]) > 1e-5: raise ValueError('Initial token probe replay mismatch')
        probe_comparison = {'fold':'val','rows':len(probe_rows),'initial':initial_probe,
                            'trained':molecular_probe(model,provider,probe_rows,cache),
                            'initial_parameter_hash_verified':True}
        preds,stats = molecular_predictions(model,provider,idx,folds['train'],cache,interventions)
    elif mode in BIT_MODES:
        # Transplant context queries, never entire attention rows: the active
        # bit masks belong to the receiving molecule and differ in length.
        queries = torch.cat([model.queries(torch.from_numpy(provider[idx[s:s+2048]]))
                             for s in range(0, len(idx), 2048)])
        variants = {'native': queries}
        stats = {'intervention': 'replace context queries; preserve recipient fingerprint mask, values and other branches'}
        if interventions:
            train_sample = np.random.default_rng(20260908).choice(folds['train'], min(8192, len(folds['train'])), replace=False)
            train_queries = torch.cat([model.queries(torch.from_numpy(provider[ids])) for ids in np.array_split(train_sample, 4)])
            variants['train_mean_query'] = train_queries.mean(0, keepdim=True).expand_as(queries)
            total = torch.zeros(3, model.router.key_dim, dtype=torch.float64)
            for start in range(0, len(folds['train']), 2048):
                ids = folds['train'][start:start+2048]
                total += model.queries(torch.from_numpy(provider[ids])).double().sum(0)
            exact_mean = (total/len(folds['train'])).to(queries.dtype)
            variants['full_train_mean_query'] = exact_mean[None].expand_as(queries)
            stats['sampled_mean_query_training_rows'] = len(train_sample)
            stats['full_mean_query_training_rows'] = len(folds['train'])
            stats['full_training_mean_query'] = exact_mean.tolist()
            variants['uniform'] = queries
            for seed in (101, 102, 103):
                variants[f'shuffle_query_{seed}'] = queries[np.random.default_rng(seed).permutation(len(idx))]
        for name, q in variants.items():
            blocks = []
            for start in range(0, len(idx), 256):
                x = torch.from_numpy(provider[idx[start:start+256]])
                o = model.forward_with_queries(x, q[start:start+256], uniform=name == 'uniform')
                blocks.append(np.column_stack([o[LABEL_TO_HEAD[c]].numpy() for c in range(7)]))
            preds[name] = np.concatenate(blocks)
    elif routed:
        tokens = torch.cat([model.tokens(torch.from_numpy(provider[idx[s:s+2048]]))
                            for s in range(0, len(idx), 2048)])
        routes = torch.cat([model.routes(t) for t in tokens.split(2048)])
        stats = {'per_molecule_sd': routes.std(0).mean().item(),
                 'routing_axes': (['head', 'query_token', 'key_token'] if mode in SELF_MODES
                                  else ['channel', 'slot', 'key_token']),
                 'query_variation': routes.std(2).mean().item(),
                 'entropy': -(routes * routes.clamp_min(1e-9).log()).sum(-1).mean().item(),
                 'mean_weights': routes.mean(0).tolist()}
        variants = {'native': routes}
        if interventions:
            train_sample = np.random.default_rng(20260908).choice(folds['train'],
                                     min(8192, len(folds['train'])), replace=False)
            train_routes = torch.cat([model.routes(model.tokens(torch.from_numpy(provider[ids])))
                                      for ids in np.array_split(train_sample, 4)])
            variants['train_mean'] = train_routes.mean(0, keepdim=True).expand_as(routes)
            variants['uniform'] = torch.full_like(routes, 1/routes.shape[-1])
            for seed in (101, 102, 103):
                order = np.random.default_rng(seed).permutation(len(idx))
                variants[f'shuffle_{seed}'] = routes[order]
        for name, w in variants.items():
            blocks = []
            for start in range(0, len(idx), 2048):
                o = model.from_tokens(tokens[start:start+2048], w[start:start+2048])
                blocks.append(np.column_stack([o[LABEL_TO_HEAD[c]].numpy() for c in range(7)]))
            preds[name] = np.concatenate(blocks)
    else:
        blocks = []
        for start in range(0, len(idx), 2048):
            o = model(torch.from_numpy(provider[idx[start:start+2048]]))
            blocks.append(np.column_stack([o[LABEL_TO_HEAD[c]].numpy() for c in range(7)]))
        preds['native'] = np.concatenate(blocks)
    for p in preds.values():
        for c, (mean, std) in ck['reg_scalers'].items():
            p[:, c] = p[:, c] * std + mean
    path = run / f'predictions_{fold}.npz'
    np.savez_compressed(path, idx=idx, labels=y, **preds)
    result = {'mode': mode, 'seed': ck['seed'], 'fold': fold, 'epoch': ck['epoch'],
              'loss': ck.get('loss', 'legacy'), 'selection': ck.get('selection', 'spearman'),
              'descriptor_transform': ck.get('descriptor_transform', 'none'),
              'chemberta_cache': ck.get('chemberta_cache'),
              'token_cache': ck.get('token_cache'),
              'packed_tokens': ck.get('packed_tokens', False),
              'cav_label_fraction': ck.get('cav_label_fraction', 1.), 'label_budget':ck.get('label_budget'),
              'parameters': model.param_report(), 'routing': stats,
              'matched_routing_probe': probe_comparison,
              'metrics': {k: metrics(y, p) for k, p in preds.items()}}
    (run / f'evaluation_{fold}.json').write_text(json.dumps(result, indent=2))
    return idx, y, preds, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True, nargs='+', help='one parent directory per arm')
    ap.add_argument('--fold', choices=['val', 'test'], default='val')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    ap.add_argument('--interventions', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        ap.error('Seeds must be unique')
    torch.set_num_threads(1)
    ensembles, results, per_seed = {}, {}, {}
    ref_idx = ref_y = None
    for arm in args.runs:
        arm_name = arm.parent.name + '__' + arm.name
        if arm_name in per_seed:
            raise ValueError(f'Duplicate arm identifier: {arm_name}')
        gathered, per_seed[arm_name] = {}, []
        for seed in args.seeds:
            idx, y, predictions, result = predict_run(arm / f'seed_{seed}', args.fold, args.interventions)
            if ref_idx is not None:
                assert np.array_equal(idx, ref_idx) and np.array_equal(y, ref_y, equal_nan=True)
            ref_idx, ref_y = idx, y
            per_seed[arm_name].append(result)
            for variant, p in predictions.items():
                gathered.setdefault(variant, []).append(p)
        for variant, ps in gathered.items():
            key = arm_name + '/' + variant
            ensembles[key] = np.mean(ps, axis=0)
            results[key] = metrics(y, ensembles[key])
        suffix = '' if sorted(args.seeds) == [42,43] else '_seeds_'+'_'.join(map(str,sorted(args.seeds)))
        np.savez_compressed(arm / f'ensemble_predictions_{args.fold}{suffix}.npz', idx=idx, labels=y,
                            **{v: np.mean(ps, 0) for v, ps in gathered.items()})
        print(arm_name, results[arm_name+'/native']['wrmse'], flush=True)
    comparisons = {}
    natives = [k for k in ensembles if k.endswith('/native')]
    for i, a in enumerate(natives):
        for b in natives[:i]:
            comparisons[a+' minus '+b] = paired_bootstrap(y, ensembles[a], ensembles[b])
    for key in ensembles:
        native = key.split('/')[0]+'/native'
        if key != native:
            comparisons[key+' minus '+native] = paired_bootstrap(y, ensembles[key], ensembles[native])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'fold': args.fold, 'seeds': args.seeds, 'metrics': results,
                                   'regression_ensemble':'mean pChEMBL predictions',
                                   'classification_ensemble':'mean logits; reported AUC uses logits directly',
                                   'paired': comparisons, 'per_seed': per_seed}, indent=2))


if __name__ == '__main__':
    main()
