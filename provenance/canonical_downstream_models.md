# MOCHA Canonical Downstream Models

These are the canonical checkpoints corresponding to the reported paper results.

| Dataset | Variant | Paper mIoU | Exact metric | Epoch | Released checkpoint |
|---|---|---:|---:|---:|---|
| nuscenes | mocha | 0.81 | 0.8057 | 1 | `downstream/nuscenes/mocha/model_best.pth` |
| semantickitti | multimocha_ft | 0.59 | 0.5926 | 5 | `downstream/semantickitti/multimocha/model_best.pth` |
| semantickitti | targetmocha | 0.60 | 0.5952 | 5 | `downstream/semantickitti/targetmocha/model_best.pth` |
| toronto3d | multimocha_ft | 0.58 | 0.5842 | 5 | `downstream/toronto3d/multimocha/model_best.pth` |
| toronto3d | targetmocha | 0.59 | 0.5896 | 5 | `downstream/toronto3d/targetmocha/model_best.pth` |
| rellis3d | multimocha_ft | 0.27 | 0.2667 | 5 | `downstream/rellis3d/multimocha/model_best.pth` |
| rellis3d | targetmocha | 0.27 | 0.2678 | 5 | `downstream/rellis3d/targetmocha/model_best.pth` |
| semanticusl | multimocha_ft | 0.24 | 0.2394 | 5 | `downstream/semanticusl/multimocha/model_best.pth` |
| semanticusl | targetmocha | 0.25 | 0.2451 | 4 | `downstream/semanticusl/targetmocha/model_best.pth` |
| dales | multimocha_ft | 0.79 | 0.7872 | 19 | `downstream/dales/multimocha/model_best.pth` |
| dales | targetmocha | 0.79 | 0.7869 | 19 | `downstream/dales/targetmocha/model_best.pth` |

## Release policy

- Checkpoints are distributed through this Hugging Face repository.
- Source code, configurations, documentation, and reproducibility provenance are maintained in the companion MOCHA GitHub repository.
- Third-party datasets are not redistributed.
- Jetson deployment artifacts retain explicit parent-checkpoint provenance.
- DALES 10-view TTA is a research-evaluation setting and is not the validated edge-deployment mode.
