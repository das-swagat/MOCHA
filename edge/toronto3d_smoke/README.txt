MOCHA Jetson Stage 1 package

Purpose
-------
Prepare a minimal Toronto3D TargetMOCHA Jetson smoke-test package on Goose,
then build a reusable Jetson Docker image and run one real LiDAR inference.

Design choices
--------------
- Source H100 checkpoint remains untouched.
- Training optimizer/scheduler state is not transferred.
- TargetMOCHA occ_head is removed because source code proves it is training-only.
- Floating weights are stored in FP16 to reduce transfer/storage.
- First runtime loads them into an FP32 model and uses CUDA AMP FP16.
  This is conservative for parity; pure FP16 parameter storage is a later optimization.
- One median Toronto3D validation tile is used: L002_x0003_y0006 (12,702 raw points).
- Jetson host Python is never modified. Dependencies live only in Docker.
