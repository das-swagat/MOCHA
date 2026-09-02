#!/usr/bin/env python3
import argparse
import json
import pickle
import random
from pathlib import Path


SEMANTIC_KITTI_TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
SEMANTIC_USL_TRAIN_SEQS = ["03", "12"]


def balanced_sample(items, n, seed):
    rng = random.Random(seed)
    items = list(items)
    if not items:
        return []
    if len(items) >= n:
        return rng.sample(items, n)
    out = []
    while len(out) < n:
        tmp = items[:]
        rng.shuffle(tmp)
        out.extend(tmp)
    return out[:n]


def add_nuscenes(project_root, n_per_domain, seed):
    info_path = project_root / "02_data/nuscenes_pointcept_trainval/info/nuscenes_infos_10sweeps_train.pkl"
    raw_root = project_root / "02_data/nuscenes_raw"
    infos = pickle.load(open(info_path, "rb"))

    items = []
    for x in infos:
        p = raw_root / x["lidar_path"]
        if p.exists():
            items.append({
                "domain": "nuscenes",
                "format": "bin5",
                "path": str(p),
            })

    return balanced_sample(items, n_per_domain, seed)


def add_semantic_kitti(project_root, n_per_domain, seed):
    root = project_root / "02_data/semantic_kitti/dataset/sequences"
    items = []
    for seq in SEMANTIC_KITTI_TRAIN_SEQS:
        velo = root / seq / "velodyne"
        for p in sorted(velo.glob("*.bin")):
            items.append({
                "domain": "semantic_kitti",
                "format": "bin4",
                "path": str(p),
            })

    return balanced_sample(items, n_per_domain, seed)


def add_semantic_usl(project_root, n_per_domain, seed):
    root = project_root / "02_data/semanticusl_labeled4/dataset/sequences"
    items = []
    for seq in SEMANTIC_USL_TRAIN_SEQS:
        velo = root / seq / "velodyne"
        for p in sorted(velo.glob("*.bin")):
            items.append({
                "domain": "semantic_usl",
                "format": "bin4",
                "path": str(p),
            })

    return balanced_sample(items, n_per_domain, seed)


def add_pointcept_npy(project_root, rel_root, domain, n_per_domain, seed):
    root = project_root / rel_root / "train"
    items = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "coord.npy").exists() and (d / "strength.npy").exists():
            items.append({
                "domain": domain,
                "format": "pointcept_npy",
                "root": str(d),
            })

    return balanced_sample(items, n_per_domain, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--n-per-domain", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="06_manifests/multimocha/multimocha_mini_v1_train.jsonl")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = []
    all_items += add_nuscenes(project_root, args.n_per_domain, args.seed + 1)
    all_items += add_semantic_kitti(project_root, args.n_per_domain, args.seed + 2)
    all_items += add_semantic_usl(project_root, args.n_per_domain, args.seed + 3)
    all_items += add_pointcept_npy(project_root, "02_data/rellis3d_pointcept", "rellis3d", args.n_per_domain, args.seed + 4)
    all_items += add_pointcept_npy(project_root, "02_data/toronto3d_pointcept", "toronto3d", args.n_per_domain, args.seed + 5)

    rng = random.Random(args.seed)
    rng.shuffle(all_items)

    with open(out_path, "w") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    counts = {}
    for item in all_items:
        counts[item["domain"]] = counts.get(item["domain"], 0) + 1

    summary_path = out_path.with_suffix(".summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"manifest: {out_path}\n")
        f.write(f"total: {len(all_items)}\n")
        f.write(f"n_per_domain: {args.n_per_domain}\n")
        f.write(f"seed: {args.seed}\n")
        for k in sorted(counts):
            f.write(f"{k}: {counts[k]}\n")

    print(f"wrote: {out_path}")
    print(f"wrote: {summary_path}")
    print(f"total: {len(all_items)}")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")


if __name__ == "__main__":
    main()
