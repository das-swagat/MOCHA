#!/usr/bin/env python3
import argparse, json, sys, time
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
        u = (p | t).sum().item()
        if u:
            vals.append((p & t).sum().item() / u)
    return sum(vals)/len(vals) if vals else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    print("gpu:", torch.cuda.get_device_name(0))

    spec = json.loads((HERE/"model_spec.json").read_text())
    model = EdgeSegmentor(spec["backbone"], spec["backbone_out_channels"], spec["num_classes"])

    obj = torch.load(HERE/"targetmocha_58.96_fp16_storage.pth", map_location="cpu", weights_only=False)
    state = obj["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("missing:", missing)
        print("unexpected:", unexpected)
        raise SystemExit("state_dict mismatch")

    model = model.cuda().eval()
    sample = torch.load(HERE/"sample_preprocessed.pth", map_location="cpu", weights_only=False)
    inp = {}
    for k in ("coord", "grid_coord", "feat", "offset"):
        v = sample[k]
        inp[k] = v.cuda(non_blocking=False)
    if "segment" in sample:
        target = sample["segment"].cuda()
    else:
        target = None

    params = sum(p.numel() for p in model.parameters())
    print("parameters:", params)
    print("processed points:", int(inp["coord"].shape[0]))

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
            times.append((time.perf_counter()-t0)*1000)

    pred = logits.argmax(1)
    hist = torch.bincount(pred, minlength=spec["num_classes"]).cpu().tolist()
    print("logits:", tuple(logits.shape), logits.dtype)
    print("pred_hist:", hist)
    if target is not None and target.numel() == pred.numel():
        print("tile_mIoU:", round(miou(pred, target, spec["num_classes"]), 6))
    print("latency_ms:", [round(x, 3) for x in times])
    print("mean_ms:", round(sum(times)/len(times), 3))
    print("peak_cuda_MiB:", round(torch.cuda.max_memory_allocated()/1024**2, 1))
    print("MOCHA JETSON SMOKE: PASS")

if __name__ == "__main__":
    main()
