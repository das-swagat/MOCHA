MOCHA SemanticKITTI Jetson package

Exact final validation adapter: SemanticKITTIColorNormalDataset.
Features: XYZ + zero-filled color(3) + zero-filled normal(3).
Default benchmark sample: deterministic 18,000-point subset after exact validation preprocessing.
Runtime: FP16 storage, FP32-resident model, CUDA autocast FP16.
TargetMOCHA occupancy head is training-only and removed for deployment.
