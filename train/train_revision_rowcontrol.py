"""Joint label-row randomized control; fork of the locked trainer.
Moves missingness and all targets together within each fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch import nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.fusion_factory import FUSION_MODES, MOLECULAR_MODES, parameter_digest
from model.revision import build_revision_model as build_model, REVISION_HEADS as ALL_HEADS
from train.revision_protocol import permute_label_rows as permute_observed_labels
from train.fusion_objectives import pooled_task_term
from train.fusion_preprocessing import transform_descriptors
from train.fusion_label_budget import apply_cav_budget
from model.cardiosafe import CardioSafe  # noqa: E402
from train.features import build_provider  # noqa: E402
from train.token_cache import open_token_cache
from train.molecular_tokens import forward_rows, molecular_probe
from train.readout_contract import capture_readout  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_cardiosafe")

# Label column -> head. Sparse: most rows are NaN in most columns.
LABEL_COLUMNS = (
    "herg_pchembl",
    "herg_blocker_10um",
    "herg_blocker_1um",
    "nav15_pchembl",
    "nav15_blocker",
    "cav12_pchembl",
    "cav12_blocker",
    "iks_blocker",
)
LABEL_TO_HEAD = dict(enumerate(LABEL_COLUMNS))
REG_COLS = (0, 3, 5)
CLS_COLS = (1, 2, 4, 6, 7)
REG_HEAD_IDX = {"herg_pchembl": 0, "nav15_pchembl": 1, "cav12_pchembl": 2}
EARLY_STOP_HEADS = ("herg_pchembl", "nav15_pchembl", "cav12_pchembl")

# Mirrors the scoring weights (hERG 0.45 / CaV1.2 0.40 / NaV1.5 0.15), halved within each
# channel between its regression and its classification heads because both are reported.
# Train what you score.
COL_WEIGHT = {
    0: 0.2250,  # herg_pchembl
    1: 0.1125,  # herg_blocker_10um
    2: 0.1125,  # herg_blocker_1um
    3: 0.0750,  # nav15_pchembl
    4: 0.0750,  # nav15_blocker
    5: 0.2000,  # cav12_pchembl
    6: 0.2000,  # cav12_blocker
}
# Fixed exploratory IKs share; preserve relative primary task weights.
COL_WEIGHT = {c: .95*w for c,w in COL_WEIGHT.items()}
COL_WEIGHT[7] = .05
assert abs(sum(COL_WEIGHT.values()) - 1.0) < 1e-9
assert set(COL_WEIGHT) == set(REG_COLS) | set(CLS_COLS)

# Reported alongside, and the single number arms are compared on.
WRMSE_WEIGHT = {"herg_pchembl": 0.45, "cav12_pchembl": 0.40, "nav15_pchembl": 0.15}


class HP:
    """Every hyperparameter, in one place. Change these, not a CLI flag."""

    epochs = 100
    patience = 15
    batch_size = 256
    accum_steps = 8
    dropout = 0.3
    # sqrt(accum_steps) = 2.83 times the un-accumulated values of 1e-4 / 1e-3 / 1e-5.
    init_lr = 2.83e-4
    max_lr = 2.83e-3
    final_lr = 2.83e-5
    warmup_epochs = 3.0
    weight_decay = 1e-5
    grad_clip = 1.0
    eval_chunk = 4096


def noam_lr(step: int, steps_per_epoch: int) -> float:
    """Linear warmup to max_lr, then exponential decay to final_lr."""
    warmup_steps = int(HP.warmup_epochs * steps_per_epoch)
    total_steps = HP.epochs * steps_per_epoch
    if step < warmup_steps:
        return HP.init_lr + (HP.max_lr - HP.init_lr) * step / max(warmup_steps, 1)
    decay_steps = max(total_steps - warmup_steps, 1)
    gamma = (HP.final_lr / HP.max_lr) ** (1 / decay_steps)
    return max(HP.max_lr * gamma ** (step - warmup_steps), HP.final_lr)


def read_labels(path: Path, n_rows: int) -> np.ndarray:
    """The label CSV as a dense (n_rows, 7) float32 matrix, empty cells as NaN."""
    out = np.full((n_rows, len(LABEL_COLUMNS)), np.nan, dtype=np.float32)
    with path.open() as handle:
        for record in csv.DictReader(handle):
            row = int(record["row_idx"])
            for col, name in enumerate(LABEL_COLUMNS):
                value = record.get(name, "")
                if value:
                    out[row, col] = float(value)
    return out


def read_split(path: Path) -> dict[str, np.ndarray]:
    """train / val / test row_idx arrays from a split CSV.

    Rows marked `excluded` belong to no fold; they exist because the split builder could not
    place them without violating the Tanimoto constraint, and silently folding them into
    train is exactly the leak the constraint exists to prevent.
    """
    from collections import defaultdict

    folds: dict[str, list[int]] = defaultdict(list)
    with path.open() as handle:
        for record in csv.DictReader(handle):
            folds[record["fold"]].append(int(record["row_idx"]))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in folds.items()}


def fit_scalers(labels: np.ndarray, train_idx: np.ndarray):
    """Regression standardisation and classification pos_weight, both fit on TRAIN only.

    Fitting either on val or on the full matrix leaks the held-out distribution into the
    target scale, which would not show up as a crash -- only as a baseline quietly too good.
    """
    reg_scalers: dict[int, tuple[float, float]] = {}
    for col in REG_COLS:
        values = labels[train_idx, col]
        values = values[~np.isnan(values)]
        reg_scalers[col] = (float(values.mean()), float(values.std()) or 1.0)

    class_weights: dict[int, float] = {}
    for col in CLS_COLS:
        values = labels[train_idx, col]
        values = values[~np.isnan(values)]
        positives = float((values > 0.5).sum())
        class_weights[col] = float((len(values) - positives) / positives) if positives else 1.0
    return reg_scalers, class_weights


def fit_task_baselines(labels: np.ndarray, train_idx: np.ndarray, class_weights: dict[int, float]):
    """Each task's loss under a trivial predictor, measured on TRAIN.

    Every term then starts at 1.0, so COL_WEIGHT is the only thing setting relative
    importance and "loss below 1" means "better than trivial" for every head alike.
    """
    out: dict[int, float] = {}
    for col in REG_COLS:
        values = labels[train_idx, col]
        values = values[~np.isnan(values)]
        # targets are standardised downstream, so predicting the mean scores their variance
        out[col] = float(((values - values.mean()) ** 2).mean() / (values.std() ** 2 or 1.0))
    for col in CLS_COLS:
        values = labels[train_idx, col]
        values = values[~np.isnan(values)]
        target = torch.from_numpy((values > 0.5).astype(np.float32))
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[col]]))
        out[col] = float(loss(torch.zeros_like(target), target))
    return out


def scale_labels(labels: np.ndarray, reg_scalers: dict[int, tuple[float, float]]) -> np.ndarray:
    out = labels.astype(np.float32).copy()
    for col, (mean, std) in reg_scalers.items():
        out[:, col] = (out[:, col] - mean) / std
    return out


@torch.no_grad()
def evaluate(model, provider, labels_raw, idx, reg_scalers, device, token_cache=None) -> dict:
    """Per-head metrics on raw pChEMBL, so numbers are comparable across arms."""
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {h: [] for h in ALL_HEADS}
    chunk = 128 if token_cache is not None else HP.eval_chunk
    for start in range(0, len(idx), chunk):
        out = forward_rows(model, provider, idx[start:start+chunk], device, token_cache)
        for head in ALL_HEADS:
            chunks[head].append(out[head].cpu().numpy())
    predicted = {h: np.concatenate(v) for h, v in chunks.items()}

    metrics: dict[str, dict] = {}
    for col, head in LABEL_TO_HEAD.items():
        truth = labels_raw[idx, col]
        present = ~np.isnan(truth)
        if present.sum() < 5:
            continue
        y_true, y_pred = truth[present], predicted[head][present]
        if col in REG_COLS:
            mean, std = reg_scalers[col]
            y_pred = y_pred * std + mean
            error = y_pred - y_true
            rho = spearmanr(y_pred, y_true).statistic if np.std(y_pred) > 1e-9 else 0.0
            metrics[head] = {
                "n": int(present.sum()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "spearman_r": float(rho if rho == rho else 0.0),
                "slope": float(np.polyfit(y_true, y_pred, 1)[0]),
                "bias": float(np.mean(error)),
            }
        else:
            y_binary = (y_true > 0.5).astype(int)
            if not 0 < y_binary.sum() < len(y_binary):
                continue
            call = (y_pred > 0).astype(int)  # heads emit logits
            tp = int(((call == 1) & (y_binary == 1)).sum())
            tn = int(((call == 0) & (y_binary == 0)).sum())
            fp = int(((call == 1) & (y_binary == 0)).sum())
            fn = int(((call == 0) & (y_binary == 1)).sum())
            metrics[head] = {
                "n": int(present.sum()),
                "pos": int(y_binary.sum()),
                "auc": float(roc_auc_score(y_binary, y_pred)),
                "sens": tp / max(tp + fn, 1),
                "spec": tn / max(tn + fp, 1),
                "mcc": float(matthews_corrcoef(y_binary, call)),
            }
    return metrics


def early_stop_score(metrics: dict) -> float:
    """Mean Spearman over the three regression heads."""
    values = [metrics[h]["spearman_r"] for h in EARLY_STOP_HEADS if h in metrics]
    return float(np.mean(values)) if values else float("-inf")


def atomic_checkpoint(checkpoint, path):
    """An interrupted write must not corrupt the previous best checkpoint."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as f:
            temporary = Path(f.name)
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="tan70_v1_1", help="a stem under data/splits/")
    parser.add_argument("--labels", type=Path, default=REPO / "data/labels/labels_v2.csv")
    parser.add_argument("--features", type=Path, default=REPO / "data/features")
    parser.add_argument("--chemberta-cache", type=Path, help="Completed row-aligned encoder readout cache")
    parser.add_argument("--token-cache", type=Path, help="Completed packed frozen-token cache")
    # Smoke-test only. Caps the loop so the pipeline can be exercised end to end without
    # spending a real run; it does not change the model, so it is not a model knob.
    parser.add_argument("--epochs", type=int, default=None, help="override HP.epochs")
    parser.add_argument(
        "--no-chemberta", action="store_true",
        help="chemical-only: drop the ChemBERTa branch entirely (fingerprints + descriptors)",
    )
    parser.add_argument(
        "--y-randomize", type=int, default=None, metavar="SEED",
        help="permute the label matrix rows with this seed before ANYTHING reads it. The "
             "permutation therefore reaches train, validation and test alike, so early "
             "stopping cannot select against real labels, and every target-derived "
             "statistic (regression scalers, pos_weight, task baselines) is fitted on the "
             "permuted matrix rather than cached from a real run.",
    )
    parser.add_argument(
        "--attention", default="none", choices=["none", "static", "trunk", "centered", "gated", *FUSION_MODES, *MOLECULAR_MODES],
        help="none=concat trunk; static=fixed queries (the arm that learned a constant); "
             "trunk=queries conditioned on the shared trunk",
    )
    parser.add_argument("--selection", choices=["spearman", "wrmse"], default="wrmse")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--loss", choices=["legacy", "pooled"], default="legacy")
    parser.add_argument("--descriptor-transform", choices=["none", "signed_log"], default="none")
    parser.add_argument('--cav-label-fraction', type=float, default=1.)
    parser.add_argument('--label-budget-seed', type=int, default=20260908)
    parser.add_argument("--test", action="store_true", help="explicitly unlock test evaluation")
    parser.add_argument("--packed-tokens", action="store_true", help="Project only valid tokens; same molecular architecture")
    args = parser.parse_args()
    if args.packed_tokens and args.attention not in MOLECULAR_MODES:
        parser.error("--packed-tokens requires a molecular-token mode")
    if (args.attention in MOLECULAR_MODES) != (args.token_cache is not None):
        parser.error("Molecular modes require --token-cache; other modes must omit it")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'result.json').exists():
        parser.error('Refusing to overwrite a completed run; choose a new --out directory')
    source_paths = sorted([*REPO.glob('model/*.py'), *REPO.glob('train/*.py')])
    provenance = {
        'captured_at_start_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'arguments': vars(args),
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(),
        'thread_environment': {k: os.environ.get(k) for k in ('TORCH_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str))
    if args.epochs is not None:
        HP.epochs = args.epochs
        log.warning("SMOKE RUN: epochs capped at %d", args.epochs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = read_split(REPO / "data/splits" / f"{args.split}.csv")
    train_idx, val_idx, test_idx = folds["train"], folds["val"], folds["test"]

    n_rows = sum(1 for _ in open(REPO / "data/compounds/compounds.csv")) - 1
    labels_raw = read_labels(args.labels, n_rows)
    unpermuted_labels = labels_raw.copy()
    if args.y_randomize is not None:
        labels_raw, permutation_audit = permute_observed_labels(labels_raw, folds, args.y_randomize)
        (args.out/'permutation_audit.json').write_text(json.dumps(permutation_audit, indent=2))
        log.warning('Y-RANDOMIZED whole label rows within train/val/test; %s', permutation_audit)

    labels_raw, budget_rows, label_budget = apply_cav_budget(
        labels_raw, train_idx, args.cav_label_fraction, args.label_budget_seed)
    np.save(args.out/'retained_cav_training_rows.npy', budget_rows)
    (args.out/'label_budget.json').write_text(json.dumps(label_budget, indent=2))
    log.info('CaV label budget %s', label_budget)

    token_cache, token_contract = open_token_cache(args.token_cache, REPO)
    readout_contract = capture_readout(args.chemberta_cache, REPO)
    if args.chemberta_cache is not None:
        cache_manifest = json.loads((args.chemberta_cache.parent/'manifest.json').read_text())
        cache_split = cache_manifest['contract'].get('split')
        if cache_split is not None and cache_split != args.split:
            raise ValueError('Predicted-property scaling belongs to a different split')
    descriptors = np.load(args.features / "descriptors_20.npy").astype(np.float32)
    desc_mean = descriptors[train_idx].mean(axis=0)
    desc_std = descriptors[train_idx].std(axis=0)
    desc_std[desc_std < 1e-8] = 1.0
    provider = build_provider(args.features, transform_descriptors(
        (descriptors - desc_mean) / desc_std, args.descriptor_transform),
        chemberta_path=args.chemberta_cache)

    log.info(
        "split %s  features %s  train/val/test %d/%d/%d",
        args.split, provider.shape, len(train_idx), len(val_idx), len(test_idx),
    )

    reg_scalers, class_weights = fit_scalers(labels_raw, train_idx)
    baselines = fit_task_baselines(labels_raw, train_idx, class_weights)
    labels = scale_labels(labels_raw, reg_scalers)
    log.info("task baselines %s", {LABEL_TO_HEAD[c]: round(v, 4) for c, v in baselines.items()})

    model = build_model(mode=args.attention, slots=args.slots,
                        chem_dim=provider.chem_dim, dropout=HP.dropout, encoder_dim=provider.encoder_dim,
                        chemberta=not args.no_chemberta, packed_tokens=args.packed_tokens).to(device)
    provenance['initial_parameter_sha256'] = parameter_digest(model)
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str))
    for col, (mean, std) in reg_scalers.items():
        i = REG_HEAD_IDX[LABEL_TO_HEAD[col]]
        model.reg_mean[i], model.reg_std[i] = mean, std
    log.info(
        "model chem_dim=%d attention=%s params=%s (attn %s)",
        provider.chem_dim, args.attention, f"{model.param_report()['total']:,}",
        f"{model.param_report()['attention']:,}",
    )

    model.eval()
    probe = torch.from_numpy(provider[val_idx[:4096]]).float().to(device)
    initial_routing = (molecular_probe(model, provider, val_idx[:1024], token_cache, device)
                       if token_cache is not None else model.attention_stats(probe) or model.gate_stats(probe))
    (args.out / "initial_routing.json").write_text(json.dumps(initial_routing, indent=2))

    bce = {
        col: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[col]], device=device))
        for col in CLS_COLS
    }
    opt = torch.optim.AdamW(model.parameters(), lr=HP.init_lr, weight_decay=HP.weight_decay)
    steps_per_epoch = int(np.ceil(len(train_idx) / (HP.batch_size * HP.accum_steps)))
    log.info(
        "accum x%d -> effective batch %d, %d optimiser steps/epoch",
        HP.accum_steps, HP.batch_size * HP.accum_steps, steps_per_epoch,
    )

    step = 0
    best, best_epoch = float("-inf"), -1
    trajectory: list[dict] = []

    def optimiser_step() -> None:
        nonlocal step
        for group in opt.param_groups:
            group["lr"] = noam_lr(step, steps_per_epoch)
        nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip)
        opt.step()
        opt.zero_grad()
        step += 1

    for epoch in range(HP.epochs):
        model.train()
        order = np.random.permutation(train_idx)
        started, running, n_micro, pending = time.time(), 0.0, 0, 0
        opt.zero_grad()
        for start in range(0, len(order), HP.batch_size):
            if args.loss == "pooled" and start % (HP.batch_size * HP.accum_steps) == 0:
                update_rows = order[start:start + HP.batch_size * HP.accum_steps]
                update_counts = np.isfinite(labels[update_rows]).sum(axis=0)
            batch = np.sort(order[start : start + HP.batch_size])
            if len(batch) < 2:  # BatchNorm needs more than one row
                continue
            y = torch.from_numpy(labels[batch]).float().to(device)
            if args.y_randomize is not None and epoch == 0 and start == 0:
                expected = scale_labels(labels_raw[batch], reg_scalers)
                np.testing.assert_array_equal(y.cpu().numpy(), expected)
                both = np.isfinite(unpermuted_labels[batch]) & np.isfinite(labels_raw[batch])
                audit = dict(batch_rows=len(batch), observed_cells=int(both.sum()),
                             changed_raw_cells=int(np.sum(unpermuted_labels[batch][both] != labels_raw[batch][both])),
                             loss_tensor_matches_permuted_scaled_labels=True)
                (args.out/'loss_tensor_audit.json').write_text(json.dumps(audit, indent=2))
            out = forward_rows(model, provider, batch, device, token_cache)

            terms = []
            for col, head in LABEL_TO_HEAD.items():
                target = y[:, col]
                present = ~torch.isnan(target)
                if not present.any():
                    continue
                predicted, truth = out[head][present], target[present]
                raw = (
                    nn.functional.mse_loss(predicted, truth)
                    if col in REG_COLS
                    else bce[col](predicted, (truth > 0.5).float())
                )
                if args.loss == "pooled":
                    raw = pooled_task_term(raw, int(present.sum()), int(update_counts[col]))
                terms.append(COL_WEIGHT[col] * raw / baselines[col])
            if not terms:
                continue
            # sum, not mean: the weights already sum to 1, and a column missing from this
            # batch must contribute nothing rather than inflate everyone else's share.
            loss = torch.stack(terms).sum()
            (loss / HP.accum_steps if args.loss == "legacy" else loss).backward()
            running += loss.item() * (HP.accum_steps if args.loss == "pooled" else 1)
            n_micro += 1
            pending += 1
            if pending == HP.accum_steps:
                optimiser_step()
                pending = 0
        if pending:  # the epoch's tail, rather than silently dropping its gradient
            optimiser_step()

        val_metrics = evaluate(model, provider, labels_raw, val_idx, reg_scalers, device, token_cache)
        val_wrmse = sum(w * val_metrics[h]["rmse"] for h, w in WRMSE_WEIGHT.items())
        score = -val_wrmse if args.selection == "wrmse" else early_stop_score(val_metrics)
        trajectory.append(
            {"epoch": epoch, "loss": running / max(n_micro, 1), "es": score,
             "seconds": round(time.time() - started, 1), "val_wrmse": val_wrmse}
        )
        (args.out / "trajectory.json").write_text(json.dumps(trajectory, default=float))
        log.info(
            "epoch %3d  loss %.4f  lr %.2e  es %.4f  %.0fs%s",
            epoch, running / max(n_micro, 1), opt.param_groups[0]["lr"], score,
            time.time() - started, "  *best" if score > best else "",
        )

        if score > best:
            best, best_epoch = score, epoch
            atomic_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "reg_scalers": reg_scalers,
                    "class_weights": class_weights,
                    "descriptor_scaler_means": desc_mean,
                    "descriptor_scaler_stds": desc_std,
                    "epoch": epoch,
                    "val_es": score,
                    "seed": args.seed,
                    "attention": args.attention,
                    "slots": args.slots,
                    "selection": args.selection,
                    "loss": args.loss,
                    "descriptor_transform": args.descriptor_transform,
                    "cav_label_fraction": args.cav_label_fraction,
                    "label_budget": label_budget,
                    "chemberta": not args.no_chemberta,
                    "chemberta_cache": readout_contract,
                    "encoder_dim": provider.encoder_dim,
                    "token_cache": token_contract,
                "packed_tokens": args.packed_tokens,
                    "y_randomize": args.y_randomize,
                    "revision_version": 1,
                    "permutation_protocol": "joint_label_rows_within_fold",
                    "task_weights": COL_WEIGHT,
                    "split": args.split,
                    "labels": str(args.labels.name),
                    "hp": {k: v for k, v in vars(HP).items() if not k.startswith("_")},
                },
                args.out / "model.pt",
            )
        elif epoch - best_epoch >= HP.patience:
            log.info("early stop: %d epochs without improvement", HP.patience)
            break

    checkpoint = torch.load(args.out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_idx = test_idx if args.test else val_idx
    test = evaluate(model, provider, labels_raw, eval_idx, reg_scalers, device, token_cache)
    # Weight variation is descriptive. The separate evaluator measures predictive
    # utility against fixed-mean/shuffled routing on the selected evaluation fold.
    model.eval()
    _probe = torch.from_numpy(provider[eval_idx[:4096]]).float().to(device)
    attn_stats = (molecular_probe(model, provider, eval_idx[:1024], token_cache, device)
                  if token_cache is not None else model.attention_stats(_probe) or model.gate_stats(_probe))
    if attn_stats:
        log.info("attention %s", attn_stats)
    wrmse = sum(w * test[h]["rmse"] for h, w in WRMSE_WEIGHT.items() if h in test)

    log.info("BEST epoch %d  val_es %.4f", best_epoch, best)
    log.info("%s wRMSE %.4f", "TEST" if args.test else "VAL", wrmse)
    for head, values in test.items():
        log.info("  %-20s %s", head, {k: round(v, 4) for k, v in values.items() if k != "n"})

    (args.out / "result.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "fold": "test" if args.test else "val",
                "selection": args.selection,
                "loss": args.loss,
                "descriptor_transform": args.descriptor_transform,
                "cav_label_fraction": args.cav_label_fraction,
                "label_budget": label_budget,
                "slots": args.slots,
                "initial_routing": initial_routing,
                "attention": args.attention,
                "chemberta": not args.no_chemberta,
                "chemberta_cache": readout_contract,
                    "encoder_dim": provider.encoder_dim,
                "token_cache": token_contract,
                "packed_tokens": args.packed_tokens,
                "y_randomize": args.y_randomize,
                    "revision_version": 1,
                    "permutation_protocol": "joint_label_rows_within_fold",
                    "task_weights": COL_WEIGHT,
                "attention_stats": attn_stats,
                "split": args.split,
                "labels": args.labels.name,
                "best_epoch": best_epoch,
                "val_es": best,
                "wrmse": wrmse,
                "metrics": test,
                "trajectory": trajectory,
            },
            indent=2,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
