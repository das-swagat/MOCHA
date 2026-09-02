MOCHA SemanticUSL Jetson smoke/benchmark package

Models
------
Baseline fine-tune: paper mIoU 23.94
TargetMOCHA: paper mIoU 24.51

Input
-----
Exact final dataset class: SemanticUSLLabeled4ColorNormalDataset.
Feature layout: XYZ coordinate + 3-channel color + 3-channel normal (9 channels).
The bundled sample is produced by the exact final validation loader.
Validation preprocessing: fixed RandomScale(0.2), GridSample(0.01),
NormalizeColor, ToTensor, Collect with feat_keys=(coord, color, normal).

Default benchmark input
-----------------------
A deterministic 18,000-point subset (or entire scan if smaller) from an
exactly preprocessed SemanticUSL validation sample. This is an initial
default; the largest tested stable Jetson count is locked after device validation.

Runtime
-------
FP16 storage only; FP32-resident model; CUDA autocast FP16.
TargetMOCHA occupancy head is training-only and removed from deployment state.
No TTA is used for the Jetson benchmark.


JETSON_VALIDATED_OPERATING_POINT
--------------------------------
Hardware: NVIDIA Jetson Orin Nano Super
Power mode: 25W
Runtime: FP32-resident model with CUDA autocast FP16
Canonical benchmark input: 5,000 points
Largest tested stable input: 5,000 points
Smallest tested unstable input: 6,000 points

Baseline:
  median latency: 1353.401 ms
  throughput: 3694.4 points/s
  peak CUDA allocation: 1109.7 MiB

TargetMOCHA:
  median latency: 1364.154 ms
  throughput: 3665.3 points/s
  peak CUDA allocation: 1109.7 MiB

TargetMOCHA's occupancy head is training-only and is not included in
deployment inference. Both deployed variants therefore have the same
157,735,099 parameter count.
