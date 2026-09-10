"""Nearest-neighbour read-across on Morgan Tanimoto -- the floor any model must clear.

For each test compound, find its k most similar training compounds by Morgan Tanimoto and
predict a similarity-weighted mean of their labels. No training, no parameters, no fitting
of any kind. It is the answer to "how much of this is chemistry the model learned, and how
much is just the nearest labelled neighbour?"

This baseline matters more than usual here because the applicability-domain behaviour
already tracks nearest-neighbour similarity: performance degrades as compounds get further
from the training set, which is what a read-across floor would predict. If a trained model
does not clear this line by a comfortable margin on a Tanimoto-controlled split, it has
learned retrieval rather than pharmacology.

A neighbour contributes only if its similarity is at least `--min-sim`. Test compounds with
no neighbour above that floor are predicted at the training prior and flagged, rather than
being handed an average over irrelevant chemistry -- on a split built to guarantee no test
compound has a training neighbour at or above the cutoff, that fraction is the point.

Tanimoto is computed as intersection over union with the intersection from a matrix product
on the binary vectors, which is exact for binary input and turns 1.1e10 pairwise
comparisons into a few BLAS calls.

USAGE
-----
    python -m train.read_across --split tan70_v1_1 --k 10
"""

from __future__ import annotations

import argparse
import csv as csvmod
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train.score_ensemble import (  # noqa: E402
    bootstrap_ci,
    classification_metrics,
    regression_metrics,
)
from train.train_cardiosafe import (  # noqa: E402
    LABEL_TO_HEAD,
    REG_COLS,
    WRMSE_WEIGHT,
    read_labels,
    read_split,
)

CHUNK = 512


def tanimoto_topk(query: np.ndarray, reference: np.ndarray, k: int):
    """Top-k Tanimoto similarities and their indices, query rows against reference rows."""
    ref_bits = reference.sum(axis=1)
    sims = np.empty((len(query), k), dtype=np.float32)
    idxs = np.empty((len(query), k), dtype=np.int64)
    for start in range(0, len(query), CHUNK):
        block = query[start : start + CHUNK]
        intersection = block @ reference.T
        union = block.sum(axis=1)[:, None] + ref_bits[None, :] - intersection
        similarity = intersection / np.maximum(union, 1.0)
        top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(len(block))[:, None]
        vals = similarity[rows, top]
        order = np.argsort(-vals, axis=1)
        idxs[start : start + len(block)] = top[rows, order]
        sims[start : start + len(block)] = vals[rows, order]
        print(f"    {min(start + CHUNK, len(query)):,}/{len(query):,}", end="\r", flush=True)
    print()
    return sims, idxs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="tan70_v1_1")
    parser.add_argument("--labels", type=Path, default=REPO / "data/labels/labels_v2.csv")
    parser.add_argument("--features", type=Path, default=REPO / "data/features")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-sim", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=REPO / "reports")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folds = read_split(REPO / "data/splits" / f"{args.split}.csv")
    train_idx, test_idx = folds["train"], folds["test"]
    n_rows = sum(1 for _ in open(REPO / "data/compounds/compounds.csv")) - 1
    labels = read_labels(args.labels, n_rows)

    morgan = np.load(args.features / "morgan_fps_2048.npy", mmap_mode="r")
    print(f"  gathering fingerprints for {len(train_idx):,} train / {len(test_idx):,} test")
    reference = np.ascontiguousarray(morgan[train_idx], dtype=np.float32)
    query = np.ascontiguousarray(morgan[test_idx], dtype=np.float32)

    print(f"  top-{args.k} Tanimoto")
    sims, idxs = tanimoto_topk(query, reference, args.k)
    top1 = sims[:, 0]
    print(f"  nearest-neighbour similarity: median {np.median(top1):.3f}, "
          f"max {top1.max():.3f}, fraction >= 0.70: {(top1 >= 0.70).mean():.4f}")

    rows: list[dict] = []
    point_rmse: dict[str, float] = {}
    for col, head in LABEL_TO_HEAD.items():
        train_labels = labels[train_idx, col]
        prior_mask = ~np.isnan(train_labels)
        if prior_mask.sum() < 5:
            continue
        prior = float(train_labels[prior_mask].mean())

        neighbour_labels = train_labels[idxs]
        weights = np.where(np.isnan(neighbour_labels), 0.0, sims)
        weights = np.where(weights >= args.min_sim, weights, 0.0)
        totals = weights.sum(axis=1)
        values = np.nan_to_num(neighbour_labels, nan=0.0)
        predicted = np.where(totals > 0, (weights * values).sum(axis=1) / np.maximum(totals, 1e-9), prior)

        truth = labels[test_idx, col]
        present = ~np.isnan(truth)
        if present.sum() < 5:
            continue
        y_true, y_pred = truth[present], predicted[present]
        no_neighbour = float((totals[present] == 0).mean())

        if col in REG_COLS:
            stats = bootstrap_ci(regression_metrics, y_true, y_pred)
            point_rmse[head] = stats["rmse"][0]
        else:
            y_binary = (y_true > 0.5).astype(int)
            if not 0 < y_binary.sum() < len(y_binary):
                continue
            # classification_metrics thresholds at 0 for logits; read-across emits a
            # probability, so centre it to keep the same decision rule at p = 0.5
            stats = bootstrap_ci(classification_metrics, y_binary, y_pred - 0.5)
        for metric, (point, lo, hi) in stats.items():
            rows.append({
                "model": f"read_across_k{args.k}", "head": head, "split": args.split,
                "fold": "test", "metric": metric, "point": round(point, 6),
                "ci_lower": round(lo, 6) if lo == lo else "",
                "ci_upper": round(hi, 6) if hi == hi else "", "n": int(present.sum()),
            })
        rows.append({
            "model": f"read_across_k{args.k}", "head": head, "split": args.split,
            "fold": "test", "metric": "frac_no_neighbour_above_min_sim",
            "point": round(no_neighbour, 6), "ci_lower": "", "ci_upper": "",
            "n": int(present.sum()),
        })

    wrmse = sum(w * point_rmse[h] for h, w in WRMSE_WEIGHT.items() if h in point_rmse)
    rows.append({
        "model": f"read_across_k{args.k}", "head": "ALL", "split": args.split, "fold": "test",
        "metric": "wrmse", "point": round(wrmse, 6), "ci_lower": "", "ci_upper": "",
        "n": int(len(test_idx)),
    })

    print(f"\nread-across k={args.k}: wRMSE {wrmse:.4f}")
    for row in rows:
        if row["metric"] in ("auc_roc", "rmse"):
            print(f"  {row['head']:20} {row['metric']:8} {row['point']:.4f} "
                  f"[{row['ci_lower']}, {row['ci_upper']}]")

    path = args.out / f"phase3_read_across_{args.split}.csv"
    with path.open("w", newline="") as handle:
        writer = csvmod.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
