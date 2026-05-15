"""Publish the CardioSafe Gradio demo to a HuggingFace Space.

Idempotent. Re-running just re-uploads the staged files; HuggingFace
diffs by content hash and `create_repo(..., exist_ok=True)` is a no-op
on an existing Space.

The Space ships **only** `README.md`, `app.py`, and `requirements.txt`.
The CardioSafe source and MolGpKa source are git-cloned by `app.py` at
Space startup. Weight files come from the CardioSafe-benchmark GitHub
Releases on first inference call (cached on the Space's persistent disk).

Usage:
  python scripts/publish_space_to_hf.py                 # full publish
  python scripts/publish_space_to_hf.py --dry-run       # stage only
  python scripts/publish_space_to_hf.py --staging PATH  # custom staging dir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGING = REPO_ROOT / "build" / "hf_space_staging"
REPO_ID = "appliedscientific/cardiosafe"
SPACE_SDK = "gradio"

REQUIRED_FILES = ("README.md", "app.py", "requirements.txt")


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


def check_staging(staging: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (staging / f).exists()]
    if missing:
        raise RuntimeError(
            f"Staging dir {staging} is missing required file(s): {missing}. "
            f"Required: {list(REQUIRED_FILES)}."
        )
    n = sum(1 for p in staging.rglob("*") if p.is_file())
    kb = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()) / 1024
    print(f"Staging ready: {n} files, {kb:.1f} KB")


def upload(staging: Path, token: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Ensuring Space {REPO_ID} (sdk={SPACE_SDK}, private={private})")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk=SPACE_SDK,
        exist_ok=True,
        private=private,
    )
    print(f"Uploading {staging} -> {REPO_ID} (Space)")
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(staging),
        repo_type="space",
        commit_message="Publish CardioSafe Gradio demo",
    )
    print(f"Done. https://huggingface.co/spaces/{REPO_ID}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate staging, do not create or upload to the Space.")
    args = p.parse_args()

    if not args.staging.exists():
        sys.exit(f"Staging dir does not exist: {args.staging}")
    check_staging(args.staging)

    if args.dry_run:
        print("--dry-run set, skipping upload.")
        return 0

    token = load_hf_token()
    upload(args.staging, token, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
