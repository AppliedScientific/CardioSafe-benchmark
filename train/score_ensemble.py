"""Score a multi-seed ensemble on a split's test fold, with bootstrap confidence intervals.

Averaging happens on the head outputs, before any inverse transform or sigmoid: the
regression heads are all standardised by the same train-fold scalers, so their outputs are
directly commensurate, and averaging logits rather than probabilities keeps a confident
minority from being flattened by a saturating sigmoid.

PROTOCOL CONSTANTS, fixed so every number stays comparable across arms:
  seeds 42-46, ensembled; B = 1000 bootstrap resamples at seed 20260428; regression metrics
  on the unscaled pChEMBL scale, inverse-transformed with the training-split scaler before
  any metric is computed; a head is scored only if it has at least 5 test compounds.

Prevalence is reported beside every AUPRC. At single-digit positive rates an AUPRC is
uninterpretable without its no-skill baseline, and the two are easy to separate later.

The single number arms are compared on is wRMSE = 0.45 hERG + 0.40 CaV1.2 + 0.15 NaV1.5
over the three regression RMSEs. It ignores the four classification heads by construction,
so read the per-head table beside it rather than instead of it.

USAGE
-----
    python -m train.score_ensemble --runs runs --split tan70_v1_1 --label my_arm
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.cardiosafe import ALL_HEADS, CardioSafe  # noqa: E402
from train.features import build_provider  # noqa: E402
from train.train_cardiosafe import (  # noqa: E402
    LABEL_TO_HEAD,
    REG_COLS,
    WRMSE_WEIGHT,
    read_labels,
    read_split,
)

EVAL_CHUNK = 4096
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260428
MIN_TEST_COMPOUNDS = 5


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    out = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_r": float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 1e-9 else 0.0,
    }
    rho = spearmanr(y_pred, y_true).statistic if np.std(y_pred) > 1e-9 else 0.0
    out["spearman_rho"] = float(rho if rho == rho else 0.0)
    return out


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    call = (y_score > 0).astype(int)  # heads emit logits
    tp = int(((call == 1) & (y_true == 1)).sum())
    tn = int(((call == 0) & (y_true == 0)).sum())
    fp = int(((call == 1) & (y_true == 0)).sum())
    fn = int(((call == 0) & (y_true == 1)).sum())
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "prevalence": float(y_true.mean()),
        "mcc": float(matthews_corrcoef(y_true, call)),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(len(y_true), 1),
    }


def bootstrap_ci(fn, y_true, y_pred, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile CIs over `n_boot` resamples of the test indices.

    Resamples that lose a class entirely are skipped rather than scored: a single-class
    resample makes AUC undefined, and substituting 0.5 would drag the interval toward
    chance for reasons that have nothing to do with the model.
    """
    point = fn(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.dtype.kind in "iu" and len(np.unique(yt)) < 2:
            continue
        try:
            for k, v in fn(yt, y_pred[idx]).items():
                collected[k].append(v)
        except ValueError:
            continue
    out = {}
    for k, v in point.items():
        samples = np.asarray(collected[k], dtype=float)
        if samples.size:
            out[k] = (v, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        else:
            out[k] = (v, float("nan"), float("nan"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="dir holding seed_*/model.pt")
    parser.add_argument("--split", default="tan70_v1_1")
    parser.add_argument("--labels", type=Path, default=REPO / "data/labels/labels_v2.csv")
    parser.add_argument("--features", type=Path, default=REPO / "data/features")
    parser.add_argument("--fold", default="test", choices=["test", "val", "train"])
    parser.add_argument("--label", default=None, help="arm name for the CSV 'model' column")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    arm = args.label or args.runs.name

    seeds = sorted(p for p in args.runs.glob("seed_*") if (p / "model.pt").exists())
    if not seeds:
        raise SystemExit(f"no seed_*/model.pt under {args.runs}")

    device = torch.device("cpu")
    folds = read_split(REPO / "data/splits" / f"{args.split}.csv")
    idx = folds[args.fold]
    n_rows = sum(1 for _ in open(REPO / "data/compounds/compounds.csv")) - 1
    labels_raw = read_labels(args.labels, n_rows)

    first = torch.load(seeds[0] / "model.pt", map_location=device, weights_only=False)
    if first.get("y_randomize") is not None:
        # Scoring a permuted-label model against real labels would be meaningless. The
        # control has to be graded on the same permuted matrix it was trained against.
        order = np.random.default_rng(first["y_randomize"]).permutation(n_rows)
        labels_raw = labels_raw[order]
        print(f"  Y-RANDOMIZED arm: labels permuted with seed {first['y_randomize']}")

    reg_scalers = first["reg_scalers"]
    desc_mean = np.asarray(first["descriptor_scaler_means"], dtype=np.float32)
    desc_std = np.asarray(first["descriptor_scaler_stds"], dtype=np.float32)
    descriptors = np.load(args.features / "descriptors_20.npy").astype(np.float32)
    provider = build_provider(args.features, (descriptors - desc_mean) / desc_std)

    summed: dict[str, np.ndarray] = {}
    for seed_dir in seeds:
        checkpoint = torch.load(seed_dir / "model.pt", map_location=device, weights_only=False)
        # Averaging checkpoints that disagree on scalers or architecture would produce a
        # plausible number from incompatible predictions, and nothing would complain.
        if checkpoint["reg_scalers"] != reg_scalers:
            raise SystemExit(f"{seed_dir} carries different reg_scalers; refusing to average")
        for key in ("attention", "chemberta", "y_randomize"):
            if checkpoint.get(key) != first.get(key):
                raise SystemExit(f"{seed_dir} differs on {key}; refusing to mix variants")
        model = CardioSafe(
            chem_dim=provider.chem_dim,
            attention=checkpoint.get("attention", "none"),
            chemberta=checkpoint.get("chemberta", True),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        chunks: dict[str, list[np.ndarray]] = {h: [] for h in ALL_HEADS}
        with torch.no_grad():
            for start in range(0, len(idx), EVAL_CHUNK):
                batch = torch.from_numpy(provider[idx[start : start + EVAL_CHUNK]]).float()
                out = model(batch.to(device))
                for head in ALL_HEADS:
                    chunks[head].append(out[head].cpu().numpy())
        for head in ALL_HEADS:
            summed[head] = summed.get(head, 0.0) + np.concatenate(chunks[head])
        print(f"  scored {seed_dir.name}", flush=True)

    predicted = {h: v / len(seeds) for h, v in summed.items()}

    rows: list[dict] = []
    point_rmse: dict[str, float] = {}
    for col, head in LABEL_TO_HEAD.items():
        truth = labels_raw[idx, col]
        present = ~np.isnan(truth)
        if present.sum() < MIN_TEST_COMPOUNDS:
            continue
        y_true, y_pred = truth[present], predicted[head][present]
        if col in REG_COLS:
            mean, std = reg_scalers[col]
            stats = bootstrap_ci(regression_metrics, y_true, y_pred * std + mean)
            point_rmse[head] = stats["rmse"][0]
        else:
            y_binary = (y_true > 0.5).astype(int)
            if not 0 < y_binary.sum() < len(y_binary):
                continue
            stats = bootstrap_ci(classification_metrics, y_binary, y_pred)
        for metric, (point, lo, hi) in stats.items():
            rows.append(
                {
                    "model": arm,
                    "head": head,
                    "split": args.split,
                    "fold": args.fold,
                    "metric": metric,
                    "point": round(point, 6),
                    "ci_lower": round(lo, 6) if lo == lo else "",
                    "ci_upper": round(hi, 6) if hi == hi else "",
                    "n": int(present.sum()),
                }
            )

    wrmse = sum(w * point_rmse[h] for h, w in WRMSE_WEIGHT.items() if h in point_rmse)
    rows.append(
        {"model": arm, "head": "ALL", "split": args.split, "fold": args.fold,
         "metric": "wrmse", "point": round(wrmse, 6), "ci_lower": "", "ci_upper": "",
         "n": int(len(idx))}
    )

    print(f"\n{arm}: {len(seeds)}-seed ensemble, {args.split} {args.fold}, n={len(idx):,}")
    print(f"wRMSE {wrmse:.4f}\n")
    by_head: dict[str, dict] = {}
    for r in rows:
        by_head.setdefault(r["head"], {})[r["metric"]] = (r["point"], r["ci_lower"], r["ci_upper"])
    for head, metrics in by_head.items():
        if head == "ALL":
            continue
        key = "auc_roc" if "auc_roc" in metrics else "rmse"
        p, lo, hi = metrics[key]
        extra = "auprc" if key == "auc_roc" else "pearson_r"
        print(f"  {head:20} {key} {p:.4f} [{lo}, {hi}]   {extra} {metrics[extra][0]:.4f}")

    csv_path = args.csv or args.runs / f"metrics_{arm}_{args.split}_{args.fold}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csvmod.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.runs / f"ensemble_{args.split}_{args.fold}.json"
    json_path.write_text(
        json.dumps(
            {
                "model": arm,
                "seeds": [s.name for s in seeds],
                "attention": first.get("attention", "none"),
                "chemberta": first.get("chemberta", True),
                "y_randomize": first.get("y_randomize"),
                "split": args.split,
                "fold": args.fold,
                "labels": args.labels.name,
                "n": int(len(idx)),
                "wrmse": wrmse,
                "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
                "rows": rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
