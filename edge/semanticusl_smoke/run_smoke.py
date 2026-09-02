#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))

from pointcept.models.builder import build_model
import pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia
from pointcept.models.utils.structure import Point


class EdgeSegmentor(nn.Module):
    def __init__(self, backbone_cfg, backbone_out_channels, num_classes):
        super().__init__()
        self.backbone = build_model(backbone_cfg)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        return self.seg_head(feat)


def miou(pred, target, classes):
    vals = []
    for c in range(classes):
        p = pred == c
        t = target == c
        union = (p | t).sum().item()
        if union:
            vals.append((p & t).sum().item() / union)
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values, p):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    r = (len(values) - 1) * p
    lo = int(r)
    hi = min(lo + 1, len(values) - 1)
    f = r - lo
    return values[lo] * (1 - f) + values[hi] * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "targetmocha"], default="baseline")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    spec = json.loads((HERE / "model_spec.json").read_text())
    variants = json.loads((HERE / "variants.json").read_text())
    v = variants[args.variant]

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    print("gpu:", torch.cuda.get_device_name(0))
    print("dataset: SemanticUSL")
    print("variant:", args.variant)
    print("paper_mIoU:", v["paper_mIoU"])

    model = EdgeSegmentor(
        spec["backbone"],
        spec["backbone_out_channels"],
        spec["num_classes"],
    )

    obj = torch.load(HERE / v["weight"], map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(obj["state_dict"], strict=False)

    if missing or unexpected:
        print("missing:", missing)
        print("unexpected:", unexpected)
        raise SystemExit("state_dict mismatch")

    model = model.float().cuda().eval()

    sample = torch.load(
        HERE / "sample_preprocessed.pth",
        map_location="cpu",
        weights_only=False,
    )

    inp = {k: sample[k].cuda() for k in ("coord", "grid_coord", "feat", "offset")}

    target = sample.get("segment")
    if target is not None:
        target = target.cuda()

    npoints = int(inp["coord"].shape[0])

    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("processed points:", npoints)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inp)

        torch.cuda.synchronize()

        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inp)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    pred = logits.argmax(1)

    print("logits:", tuple(logits.shape), logits.dtype)
    print("pred_hist:", torch.bincount(pred, minlength=spec["num_classes"]).cpu().tolist())

    if target is not None and target.numel() == pred.numel():
        print("tile_mIoU:", round(miou(pred, target, spec["num_classes"]), 6))

    mean_ms = sum(times) / len(times)
    median_ms = statistics.median(times)
    p95_ms = percentile(times, 0.95)

    print("latency_ms:", [round(x, 3) for x in times])
    print("mean_ms:", round(mean_ms, 3))
    print("median_ms:", round(median_ms, 3))
    print("p95_ms:", round(p95_ms, 3))
    print("fps_from_median:", round(1000.0 / median_ms, 3))
    print("points_per_s_from_median:", round(npoints / (median_ms / 1000.0), 1))
    print("peak_cuda_MiB:", round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print("MOCHA SEMANTICUSL JETSON SMOKE: PASS")


if __name__ == "__main__":
    main()
