# Dataset Preparation and Preprocessing

MOCHA does not redistribute datasets. Obtain each dataset from its original
provider as described in `DATASETS.md`.

Released checkpoints are dataset-specific. Do not assume that an arbitrary
XYZ point cloud can be passed directly to a released model.

## Released model inputs

| Dataset | Model input | Test voxelization | Notes |
|---|---|---:|---|
| nuScenes | `coord + color + normal` | 0.01 m | Released MOCHA model uses 3 LiDAR sweeps |
| SemanticKITTI | `coord + color + normal` | 0.01 m | 9-channel model input |
| SemanticUSL | `coord + color + normal` | 0.01 m | 9-channel model input |
| Toronto3D | `coord + strength` | 0.02 m | 4-channel model input |
| RELLIS-3D | `coord + strength` | 0.01 m | 4-channel model input |
| DALES | `coord + strength` | 0.02 m | PointSSM-style DALES pipeline |

For the released SemanticKITTI deployment adapter, color and normal channels
are zero-filled while preserving the 9-channel interface used by the model.

## Authoritative configurations

The released configuration corresponding to each checkpoint is the
authoritative specification for:

- feature construction;
- voxelization;
- transforms and augmentation;
- class definitions and label mapping;
- sweep configuration where applicable;
- evaluation settings.

Canonical paper configurations are under:

`configs/paper/`

The Hugging Face release also stores the configuration beside each released
checkpoint.

## MultiMOCHA pretraining

The canonical MultiMOCHA-15k pretraining set contains 3,000 sampled entries
from each of:

- nuScenes
- SemanticKITTI
- Toronto3D
- RELLIS-3D
- SemanticUSL

DALES is not part of MultiMOCHA pretraining.

The manifest construction utility is:

`scripts/multimocha/build_multimocha_manifest.py`

## Edge deployment

The edge packages preserve the dataset-specific feature interface used by
their corresponding released checkpoints.

Do not substitute a different feature layout, voxel size, class mapping, or
dataset preprocessing pipeline without revalidating the model.

TargetMOCHA's occupancy/support head is used during training only and is not
required for deployed semantic inference.

See `edge/` for dataset-specific deployment files and `MODEL_ZOO.md` for the
checkpoint-to-dataset mapping.
