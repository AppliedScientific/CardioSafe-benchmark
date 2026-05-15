"""Create / update the CardioSafe v1.1.0 HuggingFace Collection.

A Collection is HF's canonical "research-release landing page" — it
groups the model, dataset, and Space under one URL on the org page so
visitors can navigate between them. Title and description follow the
VCBench v1.0.0 collection pattern at
`huggingface.co/collections/appliedscientific/...`.

Idempotent. If a collection with the same title already exists in the
org, this script `add_collection_item`s the three artifacts (HF dedups,
so re-running is a no-op). Otherwise it creates the collection first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "appliedscientific"
TITLE = "CardioSafe v1.1.0 — cardiac ion-channel safety benchmark"
DESCRIPTION = (
    "CardioSafe v1.1.0 artefacts: 5-seed v1.0+v1.1 ensemble weights + L1000 "
    "encoder, audit-clean splits + labels + supplementary, Gradio demo."
)

ITEMS = [
    {
        "item_id": "appliedscientific/cardiosafe",
        "item_type": "model",
        "note": (
            "5-seed v1.0 and v1.1 paper-snapshot ensembles + L1000 expression "
            "encoder. v1.1 (audit-clean retrain) is the recommended ensemble. "
            "CC-BY-NC-4.0."
        ),
    },
    {
        "item_id": "appliedscientific/cardiosafe-benchmark",
        "item_type": "dataset",
        "note": (
            "334,444 curated compounds × 8 sparse ion-channel labels (hERG / "
            "Nav1.5 / Cav1.2 / IKs) + four Tanimoto-controlled split configs + "
            "supplementary tables / notes / figure. CC-BY-4.0."
        ),
    },
    {
        "item_id": "appliedscientific/cardiosafe",
        "item_type": "space",
        "note": (
            "Interactive Gradio demo — paste SMILES, get pIC50 + blocker CO "
            "for the four CiPA channels. Free CPU tier."
        ),
    },
]


def load_hf_token() -> str:
    if (token := os.environ.get("HF_TOKEN")):
        return token
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN not found in environment or .env")


def find_existing_collection(api, title: str):
    """Return the slug of an existing collection with `title` in NAMESPACE, or None."""
    try:
        for c in api.list_collections(owner=NAMESPACE):
            if c.title == title:
                return c.slug
    except Exception as e:
        print(f"  warning: list_collections failed: {e!r}")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--private", action="store_true",
                   help="Create the collection private (default: public).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be created without touching HF.")
    args = p.parse_args()

    from huggingface_hub import HfApi

    if args.dry_run:
        print(f"Would create/update collection in '{NAMESPACE}':")
        print(f"  title       = {TITLE!r}")
        print(f"  description = {DESCRIPTION!r}")
        for it in ITEMS:
            print(f"  + {it['item_type']:7s}  {it['item_id']}  — {it['note'][:60]}...")
        return 0

    token = load_hf_token()
    api = HfApi(token=token)

    slug = find_existing_collection(api, TITLE)
    if slug:
        print(f"Reusing existing collection: {slug}")
    else:
        print(f"Creating new collection: {TITLE}")
        collection = api.create_collection(
            title=TITLE,
            namespace=NAMESPACE,
            description=DESCRIPTION,
            private=args.private,
        )
        slug = collection.slug
        print(f"  created: https://huggingface.co/collections/{slug}")

    for it in ITEMS:
        print(f"Adding {it['item_type']:7s} {it['item_id']}")
        try:
            api.add_collection_item(
                collection_slug=slug,
                item_id=it["item_id"],
                item_type=it["item_type"],
                note=it["note"],
                exists_ok=True,
            )
        except TypeError:
            # Older huggingface_hub versions don't accept exists_ok
            try:
                api.add_collection_item(
                    collection_slug=slug,
                    item_id=it["item_id"],
                    item_type=it["item_type"],
                    note=it["note"],
                )
            except Exception as e:
                if "already" in str(e).lower() or "exists" in str(e).lower():
                    print(f"  already present, skipping")
                else:
                    raise

    print(f"\nDone. https://huggingface.co/collections/{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
