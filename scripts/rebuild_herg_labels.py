"""Rebuild the hERG classification labels from the hERG Central primary screen.

WHY THIS EXISTS
---------------
This produced `data/labels/labels_v2.csv`, which supersedes `labels_v1.csv`.

`labels_v1.csv` derived its hERG blocker calls from hERG Central with an inverted sign.
hERG Central reports percent CHANGE in current, so inhibition is NEGATIVE and a blocker is
`value < -50`. The ingestion behind v1 tested `value > 50`, which selects the activation
artifacts instead.

The script is idempotent: rerunning it against `labels_v2.csv` reproduces `labels_v2.csv`,
because the hERG columns are recomputed from the screen rather than mutated in place.

The screen ships its own binary label, `hERG_inhib`, and it is exactly reproducible from
the raw value: across all 306,893 rows, `hERG_inhib == 1` holds if and only if
`hERG_at_10uM < -50`, with zero exceptions (blocker max -50.00002, non-blocker min
-49.99382). So the sign convention is arithmetic, not inference.

Consequence of the bug: genuine blockers are labelled safe. For a cardiac safety model
those are false negatives on exactly the compounds that matter.

WHAT THIS SCRIPT DOES
---------------------
Recomputes `herg_blocker_10um` and `herg_blocker_1um` under the curation policy described
in Methods 2.1 of the paper, rather than by flipping a comparison:

  * Standardisation. SMILES are parsed, salt-stripped, charge-neutralised, sanitised and
    regenerated as canonical SMILES, then keyed by standard InChI key -- the same recipe
    the released compound pool was built with.

  * 10 uM threshold, majority vote across evidence types, ties resolved to non-blocker:
      - the 10 uM reading votes blocker if >50% inhibition, non-blocker otherwise;
      - the 1 uM reading is a measurement "at or below 3 uM", so it may vote blocker when
        it shows >50% inhibition, and abstains otherwise (a compound that blocks at low
        concentration must block at 10 uM; the converse does not follow);
      - an exact pChEMBL, where the compound has one, votes blocker if pChEMBL > 5.0.

  * 1 uM threshold: only data recorded at exactly 1 uM contributes, so the 1 uM reading
    alone decides.

Every other column is copied through byte-identical. Nav1.5, Cav1.2, IKs and all three
regression columns are untouched -- hERG Central is hERG-only and writes no potency.

USAGE
-----
    python scripts/rebuild_herg_labels.py \
        --herg-central /path/to/herg_central.tab \
        --out data/labels/labels_v2.csv

The hERG Central primary screen is available from Harvard Dataverse; it is not
redistributed here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INHIBITION_THRESHOLD = -50.0  # percent CHANGE in current; negative is block
PCHEMBL_BLOCKER_10UM = 5.0    # IC50 < 10 uM

_largest_fragment = None
_uncharger = None


def _init_worker() -> None:
    global _largest_fragment, _uncharger
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    _largest_fragment = rdMolStandardize.LargestFragmentChooser()
    _uncharger = rdMolStandardize.Uncharger()


def _standardise(args: tuple[str, float, float]) -> tuple[str, float, float] | None:
    """SMILES -> standard InChI key, following the released pool's recipe."""
    from rdkit import Chem

    smiles, at_1um, at_10um = args
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol is None:
            return None
        inchikey = Chem.MolToInchiKey(mol)
        return (inchikey, at_1um, at_10um) if inchikey else None
    except Exception:
        return None


def read_herg_central(path: Path, workers: int) -> dict[str, tuple[float, float]]:
    """Standardised InChI key -> (percent change at 1 uM, at 10 uM).

    Where two source rows collapse onto one key, the more inhibitory 10 uM reading wins:
    a screen this size carries replicate salts and tautomers of the same compound, and
    taking the stronger reading keeps a real blocker from being averaged into silence.
    """
    rows: list[tuple[str, float, float]] = []
    with path.open() as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            try:
                rows.append(
                    (
                        record["X"].strip('"'),
                        float(record["hERG_at_1uM"]),
                        float(record["hERG_at_10uM"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    print(f"  read {len(rows):,} screen rows", file=sys.stderr)

    with Pool(workers, initializer=_init_worker) as pool:
        standardised = pool.map(_standardise, rows, chunksize=500)

    out: dict[str, tuple[float, float]] = {}
    failed = 0
    for item in standardised:
        if item is None:
            failed += 1
            continue
        inchikey, at_1um, at_10um = item
        current = out.get(inchikey)
        if current is None or at_10um < current[1]:
            out[inchikey] = (at_1um, at_10um)
    print(
        f"  standardised to {len(out):,} unique InChI keys ({failed:,} unparseable)",
        file=sys.stderr,
    )
    return out


def blocker_10um(at_1um: float, at_10um: float, pchembl: float | None) -> int:
    """Majority vote across evidence types; a tie resolves to non-blocker."""
    votes = [1 if at_10um < INHIBITION_THRESHOLD else 0]
    if at_1um < INHIBITION_THRESHOLD:
        votes.append(1)  # blocks at <=3 uM, so it blocks at 10 uM; cannot vote the other way
    if pchembl is not None:
        votes.append(1 if pchembl > PCHEMBL_BLOCKER_10UM else 0)
    return 1 if sum(votes) * 2 > len(votes) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herg-central", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=REPO / "data/labels/labels_v2.csv")
    parser.add_argument("--compounds", type=Path, default=REPO / "data/compounds/compounds.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    print("hERG Central:", file=sys.stderr)
    screen = read_herg_central(args.herg_central, args.workers)

    key_of_row: dict[int, str] = {}
    with args.compounds.open() as handle:
        for record in csv.DictReader(handle):
            key_of_row[int(record["row_idx"])] = record["inchikey"]

    with args.labels.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    changed_10, changed_1, seen, conflicts = 0, 0, 0, 0
    for row in rows:
        reading = screen.get(key_of_row.get(int(row["row_idx"]), ""))
        if reading is None:
            continue
        seen += 1
        at_1um, at_10um = reading
        pchembl = float(row["herg_pchembl"]) if row["herg_pchembl"] else None
        if pchembl is not None and (pchembl > PCHEMBL_BLOCKER_10UM) != (
            at_10um < INHIBITION_THRESHOLD
        ):
            conflicts += 1

        new_10 = blocker_10um(at_1um, at_10um, pchembl)
        if row["herg_blocker_10um"] and int(float(row["herg_blocker_10um"])) != new_10:
            changed_10 += 1
        row["herg_blocker_10um"] = str(new_10)

        # Only data recorded at exactly 1 uM contributes a vote at this threshold.
        new_1 = 1 if at_1um < INHIBITION_THRESHOLD else 0
        if row["herg_blocker_1um"] and int(float(row["herg_blocker_1um"])) != new_1:
            changed_1 += 1
        row["herg_blocker_1um"] = str(new_1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nmatched into the compound pool : {seen:,}", file=sys.stderr)
    print(f"herg_blocker_10um changed      : {changed_10:,}", file=sys.stderr)
    print(f"herg_blocker_1um changed       : {changed_1:,}", file=sys.stderr)
    print(f"pChEMBL / screen disagreements : {conflicts:,}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
