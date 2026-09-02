MOCHA RELLIS-3D Jetson smoke/benchmark package

Models
------
Baseline fine-tune: paper mIoU 26.67
TargetMOCHA: paper mIoU 26.78

Input
-----
Exact final dataset class: Pointcept DefaultDataset.
Feature layout: XYZ coordinate + scalar strength/intensity (4 channels).
Validation preprocessing: fixed RandomScale(0.2), GridSample(0.01),
ToTensor, Collect with feat_keys=(coord, strength).

Default benchmark input
-----------------------
A deterministic 18,000-point subset (or entire scan if smaller) from an
exactly preprocessed RELLIS-3D validation sample. This is an initial
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
Canonical benchmark input: 9,000 points
Largest tested stable input: 9,000 points
Smallest tested unstable input: 10,000 points

Baseline:
  median latency: 2111.600 ms
  throughput: 4262.2 points/s
  peak CUDA allocation: 1242.1 MiB

TargetMOCHA:
  median latency: 2100.243 ms
  throughput: 4285.2 points/s
  peak CUDA allocation: 1242.1 MiB

TargetMOCHA's occupancy head is training-only and is not included in
deployment inference. Both deployed variants therefore have the same
157,734,774 parameter count.
