#!/usr/bin/env python3

import argparse

from model_registry import HF_REPO, MODELS


def show_model(alias):
    if alias not in MODELS:
        print(f"Unknown model: {alias}")
        print("\nAvailable models:")
        for name in MODELS:
            print(f"  {name}")
        raise SystemExit(2)

    m = MODELS[alias]

    print("☕ MOCHA | Model Information")
    print("   Multi-scale Occupancy-Guided Cross-View Hierarchical Alignment")
    print("─" * 64)
    print(f"Model         {alias}")
    print(f"Family        {m['family']}")
    print(f"Dataset       {m['dataset']}")
    print(f"Classes       {m['classes']}")
    print(f"Input         {m['in_channels']} channels")
    print(f"Features      {m['features']}")
    print(f"Voxel grid    {m['grid_size']}")
    if m["sweeps"] > 1:
        print(f"LiDAR sweeps  {m['sweeps']}")
    print()
    print("Preprocessing")
    print(f"  {m['preprocessing']}")
    print()
    print("Hugging Face")
    print(f"  Repository  {HF_REPO}")
    print(f"  Checkpoint  {m['checkpoint']}")
    print(f"  Config      {m['config']}")


def show_table(details=False):
    print("☕ MOCHA | Available Models")
    print("   Multi-scale Occupancy-Guided Cross-View Hierarchical Alignment")
    print()

    if details:
        header = (
            f"{'MODEL':<27} {'DATASET':<15} {'FAMILY':<12} "
            f"{'INPUT':<9} {'CLASSES':>7} {'mIoU':>7}"
        )
        print(header)
        print("─" * len(header))

        for alias, m in MODELS.items():
            inp = f"{m['in_channels']} ch"
            print(
                f"{alias:<27} "
                f"{m['dataset']:<15} "
                f"{m['family']:<12} "
                f"{inp:<9} "
                f"{m['classes']:>7} "
                f"{m['paper_miou']:>7.2f}"
            )
    else:
        header = f"{'MODEL':<27} {'DATASET':<15} {'FAMILY':<12} {'INPUT':<12}"
        print(header)
        print("─" * len(header))

        for alias, m in MODELS.items():
            inp = f"{m['in_channels']} channels"
            print(
                f"{alias:<27} "
                f"{m['dataset']:<15} "
                f"{m['family']:<12} "
                f"{inp:<12}"
            )

    print()
    print("For preprocessing and checkpoint details:")
    print("  python3 tools/list_models.py --model <MODEL>")


def main():
    parser = argparse.ArgumentParser(
        description="List canonical MOCHA downstream models."
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show research metrics and technical details.",
    )
    parser.add_argument(
        "--model",
        help="Show detailed information for one model.",
    )
    args = parser.parse_args()

    if args.model:
        show_model(args.model)
    else:
        show_table(details=args.details)


if __name__ == "__main__":
    main()
