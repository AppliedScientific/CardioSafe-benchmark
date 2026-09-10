"""Explicit token batches and recipient-preserving routing interventions."""
import numpy as np
import torch


def forward_rows(model, provider, rows, device='cpu', cache=None, queries=None, uniform=False):
    x = torch.from_numpy(provider[rows]).float().to(device)
    if cache is None:
        if hasattr(model,'token_pool'):raise ValueError('Molecular model requires token states')
        return model(x)
    tokens, _, mask = cache.batch(rows)
    tokens, mask = torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device)
    routes = None
    if uniform:
        routes = mask[:,None,:].expand(-1,3,-1).to(tokens.dtype)/mask.sum(-1)[:,None,None]
    return model(x,tokens,mask,queries=queries,routing=routes)


@torch.no_grad()
def collect_queries(model, provider, rows, device='cpu'):
    return torch.cat([model.queries(torch.from_numpy(provider[rows[s:s+256]]).float().to(device))
                      for s in range(0,len(rows),256)])


@torch.no_grad()
def molecular_probe(model, provider, rows, cache, device='cpu'):
    model.eval()
    q = collect_queries(model,provider,rows,device)
    order = np.random.default_rng(101).permutation(len(rows))
    squared, count, entropies = {}, 0, []
    for s in range(0,len(rows),128):
        ids = rows[s:s+128]
        native = forward_rows(model,provider,ids,device,cache)
        for name,extra in [('uniform',{'uniform':True}),('shuffle_query_101',{'queries':q[order][s:s+128]})]:
            altered = forward_rows(model,provider,ids,device,cache,**extra)
            for h in native:
                squared.setdefault(name,{}).setdefault(h,0.)
                squared[name][h] += float((altered[h]-native[h]).double().square().sum())
        tokens, _, mask = cache.batch(ids)
        tokens, mask = torch.from_numpy(tokens).to(device),torch.from_numpy(mask).to(device)
        # Route geometry depends on recipient keys and lengths; raw cross-molecule
        # weight SD is not meaningful when token positions/lengths differ.
        x = torch.from_numpy(provider[ids]).float().to(device)
        branches,_ = model._cached_branches(x)
        w = model.token_pool.routes(torch.cat(branches[:2],1),tokens,mask)
        entropy = -(w*w.clamp_min(1e-30).log()).sum(-1)
        entropies.append((entropy/mask.sum(-1).clamp_min(2).log()[:,None]).cpu())
        count += len(ids)
    return {'rows':count,'query_per_molecule_sd':float(q.std(0).mean()),
            'mean_normalized_entropy':float(torch.cat(entropies).mean()),
            'prediction_rms_change_z_or_logit':{k:{h:(v/count)**.5 for h,v in x.items()} for k,x in squared.items()}}


@torch.no_grad()
def molecular_predictions(model, provider, idx, train_idx, cache, interventions):
    from train.train_cardiosafe import LABEL_TO_HEAD
    queries = collect_queries(model,provider,idx)
    variants = {'native':None}
    stats = {'intervention':'Replace chemical-context queries, preserving recipient tokens, masks and branch values.'}
    if interventions:
        total = torch.zeros_like(queries[0],dtype=torch.float64)
        for start in range(0,len(train_idx),2048):
            total += collect_queries(model,provider,train_idx[start:start+2048]).double().sum(0)
        mean = (total/len(train_idx)).float()
        variants['full_train_mean_query'] = mean[None].expand_as(queries)
        stats['full_mean_query_training_rows'] = len(train_idx)
        stats['full_training_mean_query'] = mean.tolist()
        variants['uniform'] = None
        for seed in (101,102,103):
            variants[f'shuffle_query_{seed}'] = queries[np.random.default_rng(seed).permutation(len(idx))]
    blocks = {k:[] for k in variants}
    # Reuse each decoded token batch across interventions to limit I/O.
    for start in range(0,len(idx),128):
        ids = idx[start:start+128]
        x = torch.from_numpy(provider[ids]).float()
        tokens,_,mask = cache.batch(ids)
        tokens,mask = torch.from_numpy(tokens),torch.from_numpy(mask)
        for name,q in variants.items():
            routing = None
            if name=='uniform':routing=mask[:,None,:].expand(-1,3,-1).float()/mask.sum(-1)[:,None,None]
            o = model(x,tokens,mask,queries=None if q is None else q[start:start+128],routing=routing)
            blocks[name].append(np.column_stack([o[LABEL_TO_HEAD[c]].numpy() for c in range(7)]))
    return {k:np.concatenate(v) for k,v in blocks.items()},stats
