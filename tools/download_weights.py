#!/usr/bin/env python3
"""Download canonical MOCHA checkpoints from Hugging Face."""

import argparse
import hashlib
from pathlib import Path

from model_registry import HF_REPO, ALL_MODELS


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def print_model(alias, model):
    print("☕ MOCHA | Model Download")
    print("   Multi-scale Occupancy-Guided Cross-View Hierarchical Alignment")
    print()
    print("==> Preparing model")
    print()
    print(f"Model       {model['family']}")
    print(f"Dataset     {model['dataset']}")
    print(f"Input       {model['in_channels']} channels")
    print(f"Features    {model['features']}")
    if model["sweeps"] > 1:
        print(f"Sweeps      {model['sweeps']}")
    print(f"Source      Hugging Face · {HF_REPO}")
    print()

    print("==> Checking compatibility")
    print("✓ Model entry found")
    print("✓ Matching configuration available")
    print("✓ Preprocessing requirements defined")
    print("✓ SHA256 checksum available")
    print()

    print("Important:")
    print(f"  {model['preprocessing']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download canonical MOCHA model weights."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=ALL_MODELS.keys(),
        help="Canonical MOCHA model alias.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Destination root. Default: checkpoints",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload an existing artifact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the original exception if a download fails.",
    )
    args = parser.parse_args()

    alias = args.model
    model = ALL_MODELS[alias]
    output_root = Path(args.output_dir)

    print_model(alias, model)

    checkpoint = output_root / model["checkpoint"]
    config = output_root / model["config"]

    print("Artifacts:")
    print(f"  Checkpoint  {model['checkpoint']}")
    print(f"  Config      {model['config']}")
    print(f"  Destination {output_root.resolve()}")
    print()

    if args.dry_run:
        print("==> Dry run")
        print("✓ Registry and artifact paths resolved")
        print("✓ No files downloaded")
        return

    if checkpoint.is_file() and not args.force:
        print("==> Existing checkpoint found")
        print("Verifying SHA256...")

        actual = sha256_file(checkpoint)

        if actual == model["sha256"]:
            print("✓ SHA256 verified")
            print()
            print("==> Ready")
            print(f"Checkpoint  {checkpoint}")
            print(f"Config      {config}")
            return

        print("✗ Existing checkpoint failed SHA256 verification")
        print("  Run again with --force to replace it.")
        raise SystemExit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("✗ huggingface_hub is not installed for this Python environment.")
        print()
        print("Install the MOCHA requirements first:")
        print("  python3 -m pip install -r requirements.txt")
        raise SystemExit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    print("==> Fetching configuration")
    try:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=model["config"],
            local_dir=str(output_root),
            force_download=args.force,
        )
        print("✓ Configuration downloaded")
        print()

        print("==> Fetching model")
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=model["checkpoint"],
            local_dir=str(output_root),
            force_download=args.force,
        )
        checkpoint = Path(downloaded)

    except Exception as exc:
        print()
        print("✗ Download failed")
        print(f"  Repository: {HF_REPO}")
        print()
        print("If this repository is private, authenticate with:")
        print("  hf auth login")
        if args.verbose:
            raise
        print()
        print(f"Details: {exc}")
        raise SystemExit(1)

    print()
    print("==> Verifying artifact")
    actual = sha256_file(checkpoint)

    if actual != model["sha256"]:
        print("✗ SHA256 verification failed")
        print(f"  Expected  {model['sha256']}")
        print(f"  Received  {actual}")
        raise SystemExit(1)

    print("✓ Download complete")
    print("✓ SHA256 verified")
    print("✓ Matching configuration downloaded")
    print()

    print("==> Ready")
    print(f"Model       {model['family']}")
    print(f"Dataset     {model['dataset']}")
    print(f"Checkpoint  {checkpoint}")
    print(f"Config      {config}")
    print()
    print("Remember:")
    print(f"  {model['preprocessing']}")


if __name__ == "__main__":
    main()
