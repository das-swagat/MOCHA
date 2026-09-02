# Edge Deployment Benchmarks

MOCHA-family semantic segmentation checkpoints were validated on an NVIDIA
Jetson Orin Nano Super Developer Kit across six LiDAR domains.

These measurements characterize embedded inference feasibility and should not
be interpreted as end-to-end real-time sensor-pipeline performance.

## Platform

| Item | Configuration |
|---|---|
| Device | NVIDIA Jetson Orin Nano Super Developer Kit, 8 GB |
| Power mode | 25 W |
| OS | Ubuntu 22.04.5 LTS |
| NVIDIA Jetson Linux / L4T | R36.4.7 |
| Kernel | Linux 5.15.148-tegra |
| CPU | 6-core Arm Cortex-A78AE |
| GPU | NVIDIA Ampere, 1024 CUDA cores, 32 Tensor Cores |
| Memory | 8 GB LPDDR5 |
| NVIDIA container release | 25.05 |
| PyTorch | 2.8.0a0+5228986c39.nv25.05 |
| CUDA reported by PyTorch | 12.9 |

## Inference policy

The released deployment configuration uses:

- FP16 checkpoint storage for reduced artifact size;
- FP32-resident model parameters at runtime;
- CUDA FP16 autocast during inference;
- `torch.inference_mode()`;
- explicit CUDA synchronization around timed inference;
- three warm-up iterations before timed measurements unless otherwise noted;
- no test-time augmentation (TTA);
- no TensorRT or INT8 quantization.

TargetMOCHA's auxiliary occupancy/support head is used during training only and
is omitted from deployed semantic-inference checkpoints.

Latency therefore measures synchronized neural-network inference after the
benchmark input has been prepared. It does not include the complete
sensor-to-prediction pipeline.

## Canonical deployment results

| Dataset | Model | Points | Params | Median (ms) | p95 (ms) | Throughput (kPts/s) | Peak CUDA (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| nuScenes | MOCHA | 4,000 | 137.276M | 1135.2 | 1140.2 | 3.52 | 1080.9 |
| SemanticKITTI | MultiMOCHA FT | 16,000 | 137.280M | 2099.8 | 2102.8 | 7.62 | 1064.3 |
| SemanticKITTI | TargetMOCHA | 16,000 | 137.280M | 2103.9 | 2106.3 | 7.61 | 1064.3 |
| Toronto3D | MultiMOCHA FT | 11,035 | 157.734M | 930.8 | 945.0 | 11.86 | 836.5 |
| Toronto3D | TargetMOCHA | 11,035 | 157.734M | 970.9 | 981.3 | 11.37 | 836.5 |
| RELLIS-3D | MultiMOCHA FT | 9,000 | 157.735M | 2111.6 | 2115.8 | 4.26 | 1242.1 |
| RELLIS-3D | TargetMOCHA | 9,000 | 157.735M | 2100.2 | 2103.8 | 4.29 | 1242.1 |
| SemanticUSL | MultiMOCHA FT | 5,000 | 157.735M | 1353.4 | 1361.4 | 3.69 | 1109.7 |
| SemanticUSL | TargetMOCHA | 5,000 | 157.735M | 1364.2 | 1367.6 | 3.67 | 1109.7 |
| DALES | MultiMOCHA FT | 18,000 | 157.734M | 688.4 | 692.9 | 26.15 | 838.3 |
| DALES | TargetMOCHA | 18,000 | 157.734M | 710.1 | 713.2 | 25.35 | 838.3 |

Toronto3D uses the final 20-run benchmark measurement. The other canonical
reported measurements use the validated package benchmark protocol.

## Validated operating points

Point count strongly affects sparse-convolution memory demand. We therefore
report dataset-specific validated operating points rather than implying that a
single point-count limit applies across datasets.

| Dataset | Packaged operating point | Additional stability observation |
|---|---:|---|
| nuScenes | 4,000 | 5,000-point test was not stable in the validated runtime |
| SemanticKITTI | 16,000 | 18,000-point test was not stable |
| Toronto3D | 11,035 | separate 25,067-point stress test also completed successfully |
| RELLIS-3D | 9,000 | 10,000-point test was not stable |
| SemanticUSL | 5,000 | 6,000-point test was not stable |
| DALES | 18,000 | original 24,068-point sample was not stable |

These thresholds are empirical observations for the released runtime and
samples, not universal hardware limits.

## Deployment interpretation

TargetMOCHA changes the fine-tuning objective but does not require its
auxiliary occupancy branch during semantic inference. Consequently, paired
MultiMOCHA and TargetMOCHA deployment variants retain the same deployed model
structure and essentially identical parameter counts within each dataset.

Small latency differences between paired variants should not be interpreted as
evidence that TargetMOCHA is inherently faster or slower.

The results demonstrate deployability on resource-constrained embedded GPU
hardware while preserving the same semantic checkpoints used for the reported
paper results.

## Reproducibility

Checkpoint-bearing Jetson smoke/benchmark packages are released through the
corresponding Hugging Face repository under `edge_packages/`.

Each package includes:

- the matching deployment runtime;
- model/configuration metadata;
- a deterministic benchmark sample;
- smoke/benchmark scripts;
- licensing and third-party notices;
- parent-checkpoint provenance.

Canonical GPU and downstream checkpoints are indexed in `MODEL_ZOO.md`.

## Limitations

These benchmarks do not measure:

- sensor acquisition latency;
- complete preprocessing and postprocessing latency;
- energy per inference;
- TensorRT performance;
- INT8 quantized performance;
- multi-stream or concurrent inference;
- hard real-time deadline guarantees.

Accordingly, MOCHA is described as having validated embedded/edge inference,
not as providing a universal real-time LiDAR segmentation guarantee.
