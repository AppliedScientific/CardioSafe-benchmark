"""Mirror the CardioSafe paper-snapshot weights to HuggingFace.

Idempotent. Re-running is safe — HuggingFace deduplicates by content hash
and `create_repo(..., exist_ok=True)` is a no-op on an existing repo.

What it does:
  1. Reads HF_TOKEN from .env (or the env var of the same name).
  2. Stages the three GitHub Releases (v1.0-weights, v1.1-weights,
     l1000-encoder-v1) into ./build/hf_staging via `gh release download`.
  3. Writes the HF model card README.md if a fresh one wasn't passed in.
  4. Creates/updates the HF model repo and uploads the staging folder.

The model-card text lives in this file (DEFAULT_README) so the script is
self-contained; pass --readme PATH to use a different file.

Usage:
  python scripts/publish_to_hf.py                       # full publish
  python scripts/publish_to_hf.py --dry-run             # stage only
  python scripts/publish_to_hf.py --skip-download       # reuse staged files
  python scripts/publish_to_hf.py --repo-id <namespace>/<name>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGING = REPO_ROOT / "build" / "hf_staging"
DEFAULT_REPO_ID = "appliedscientific/cardiosafe"

# GitHub release tag -> (staging subdir, list of asset filenames)
RELEASES = {
    "v1.0-weights": (
        "v1.0",
        [f"cardiosafe_v1.0_seed_{s}.pt" for s in (42, 43, 44, 45, 46)],
    ),
    "v1.1-weights": (
        "v1.1",
        [f"cardiosafe_v1.1_seed_{s}.pt" for s in (42, 43, 44, 45, 46)],
    ),
    "l1000-encoder-v1": (
        "l1000",
        ["l1000_encoder.pt", "l1000_per_gene_pearson.json"],
    ),
}

GH_REPO = "AppliedScientific/CardioSafe-benchmark"


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


def stage_release(tag: str, subdir: str, assets: list[str], staging: Path) -> None:
    target = staging / subdir
    target.mkdir(parents=True, exist_ok=True)
    missing = [a for a in assets if not (target / a).exists()]
    if not missing:
        print(f"  [{tag}] already staged ({len(assets)} files)")
        return
    print(f"  [{tag}] downloading {len(missing)} asset(s) into {target}")
    pattern_args: list[str] = []
    for a in missing:
        pattern_args += ["--pattern", a]
    subprocess.run(
        ["gh", "release", "download", tag, "--repo", GH_REPO, *pattern_args],
        cwd=target,
        check=True,
    )


def stage_all(staging: Path, skip_download: bool) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    print(f"Staging into {staging}")
    if skip_download:
        print("  --skip-download set, using existing staging contents")
        return
    for tag, (subdir, assets) in RELEASES.items():
        stage_release(tag, subdir, assets, staging)


def stage_readme(staging: Path, readme_path: Path | None) -> None:
    target = staging / "README.md"
    if readme_path is not None:
        shutil.copy(readme_path, target)
        print(f"  README from {readme_path} -> {target}")
        return
    if target.exists():
        print(f"  README already present at {target}")
        return
    raise RuntimeError(
        f"No README found at {target} and --readme not provided. "
        "Pass --readme PATH or drop a README.md into the staging dir."
    )


def upload(repo_id: str, staging: Path, token: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Ensuring repo {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"Uploading {staging} -> {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging),
        repo_type="model",
        commit_message="Publish CardioSafe v1.0 + v1.1 paper-snapshot weights and L1000 encoder",
    )
    print(f"Done. https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--readme", type=Path, default=None,
                        help="Path to a README.md to copy into the staging dir.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse files already in the staging dir.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage and write README, but do not create the HF repo or upload.")
    args = parser.parse_args()

    stage_all(args.staging, skip_download=args.skip_download)
    stage_readme(args.staging, args.readme)

    n_files = sum(1 for p in args.staging.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in args.staging.rglob("*") if p.is_file()) / 1e6
    print(f"Staging ready: {n_files} files, {total_mb:.1f} MB")

    if args.dry_run:
        print("--dry-run set, skipping upload.")
        return 0

    token = load_hf_token()
    upload(args.repo_id, args.staging, token, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
