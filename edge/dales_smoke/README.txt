MOCHA DALES Jetson smoke/benchmark package

Models
------
- MultiMOCHA fine-tuned: paper single-view mIoU 78.72
- TargetMOCHA: paper single-view mIoU 78.69
- DALES 10-view TTA is not used for edge deployment.

Runtime
-------
- Floating tensors are stored in FP16 only for transfer/storage reduction.
- Runtime loads them into an FP32-resident model.
- Inference uses CUDA autocast FP16.
- TargetMOCHA's occupancy head is training-only and is omitted from deployment weights.
- Jetson host Python is not modified; dependencies remain inside Docker.

Sample
------
The benchmark sample is created on Goose with the exact final DALES validation
dataset adapter and transform pipeline. The adapter maps intensity.npy to the
one-channel strength feature, so model input is XYZ + strength.

Run
---
Inside the existing MOCHA Jetson Docker image:

python3 /workspace/dales_smoke/run_smoke.py --variant multimocha --warmup 3 --runs 10
python3 /workspace/dales_smoke/run_smoke.py --variant targetmocha --warmup 3 --runs 10


Validated Jetson operating point
---------------------------------
NVIDIA Jetson Orin Nano Super, 25 W mode.
Default packaged sample: 18,000 processed DALES points.
MultiMOCHA: 688.357 ms median, 692.938 ms p95, 838.3 MiB peak CUDA.
TargetMOCHA: 710.126 ms median, 713.171 ms p95, 838.3 MiB peak CUDA.
The original 24,068-point sample was not stable under this Jetson runtime and
is retained separately as internal audit evidence, not as the package default.
