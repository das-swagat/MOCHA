MOCHA nuScenes Jetson smoke/benchmark package

Model: MOCHA sweep-3 supervised continuation, paper mIoU 80.57.
Exact final dataset class: NuScenesImagePointDataset with if_img=False.
LiDAR aggregation: if_sweep=True, sweeps=3, sweep_gap=1.
Feature layout: XYZ + color(3) + normal(3), 9 channels.
Validation preprocessing: RandomScale(0.2), GridSample(0.01), NormalizeColor, ToTensor, Collect.
Initial Jetson gate: deterministic 5,000-point subset after exact validation loading.
Runtime: FP16 storage; FP32-resident model; CUDA autocast FP16. No TTA.


JETSON_VALIDATED_OPERATING_POINT
--------------------------------
Hardware: NVIDIA Jetson Orin Nano Super
Power mode: 25W
Runtime: FP32-resident model with CUDA autocast FP16
LiDAR aggregation: 3 sweeps
Canonical benchmark input: 4,000 points
Largest tested stable input: 4,000 points
Smallest tested unstable input: 5,000 points

MOCHA:
  paper mIoU: 80.57
  parameters: 137,275,882
  mean latency: 1135.860 ms
  median latency: 1135.171 ms
  p95 latency: 1140.223 ms
  throughput: 3523.7 points/s
  peak CUDA allocation: 1080.9 MiB

The subset tile mIoU produced by run_smoke.py is a deployment
sanity-check diagnostic and is not the paper validation mIoU.
