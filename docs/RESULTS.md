# Results

Reported IoU and mIoU values use the 0-100 scale. Deltas are absolute
percentage-point changes.

## nuScenes source-domain progression

| Configuration | mIoU | App. | VRU | Bicycle | Pedestrian | Motorcycle | Construction vehicle | Traffic cone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Utonia FT continuation | 78.60 | 66.35 | 66.68 | 38.21 | 80.66 | 81.18 | 60.84 | 70.85 |
| MOCHA v2 LiDAR-only | 78.95 | 68.85 | 69.44 | 43.90 | 80.83 | 83.58 | 64.82 | 71.14 |
| + local occupancy | 79.59 | 69.33 | 69.94 | 44.11 | 81.61 | 84.11 | 64.98 | 71.82 |
| + 2-sweep continuation | 79.87 | 70.96 | 72.80 | 48.95 | 83.60 | 85.84 | 65.09 | 71.30 |
| + 3-sweep low-LR continuation | **80.57** | **72.16** | **73.93** | **52.65** | **84.44** | 84.69 | **67.35** | 71.66 |
| Delta vs. Utonia | **+1.97** | **+5.81** | **+7.24** | **+14.44** | **+3.78** | **+3.51** | **+6.51** | **+0.81** |

Final MOCHA also reaches:

- mAcc: **87.89**
- allAcc: **94.48**

App. is the application-focused mean used in the paper. VRU is the mean IoU
of bicycle, motorcycle, and pedestrian.

The improvements are class-selective rather than uniform; the paper should be
consulted for the complete classwise analysis.

## MultiMOCHA to TargetMOCHA transfer

Each comparison below uses the same canonical MultiMOCHA-15k initialization.

| Dataset | MultiMOCHA FT | TargetMOCHA | Delta |
|---|---:|---:|---:|
| SemanticKITTI | 59.26 | **59.52** | +0.26 |
| Toronto3D | 58.42 | **58.96** | +0.54 |
| RELLIS-3D | 26.67 | **26.78** | +0.11 |
| SemanticUSL | 23.94 | **24.51** | +0.57 |

TargetMOCHA adds the target-domain occupancy/support objective during
fine-tuning. Its auxiliary occupancy head is training-only and is not required
for semantic inference.

## DALES unseen aerial transfer

DALES was excluded from MultiMOCHA pretraining.

### Single-view evaluation

| Model | mIoU |
|---|---:|
| MultiMOCHA-15k FT | **78.72** |
| TargetMOCHA | 78.69 |

TargetMOCHA does not improve the single-view aggregate result.

### Matched 10-view TTA evaluation

| Model | mIoU |
|---|---:|
| MultiMOCHA-15k FT | 80.53 |
| TargetMOCHA | **80.69** |

Matched 10-view TTA improves TargetMOCHA by **+0.16** mIoU relative to the
matched MultiMOCHA evaluation.

TTA is a research-evaluation setting and is not the validated edge-deployment
configuration.

### DALES classwise change under matched TTA

| Class | MultiMOCHA | TargetMOCHA | Delta |
|---|---:|---:|---:|
| Ground | 96.69 | 96.74 | +0.05 |
| Vegetation | 93.43 | 93.52 | +0.10 |
| Cars | 84.60 | 85.55 | +0.95 |
| Trucks | 36.76 | 39.05 | +2.29 |
| Powerlines | 97.03 | 96.86 | -0.17 |
| Fences | 65.39 | 65.00 | -0.39 |
| Poles | 73.48 | 71.95 | -1.53 |
| Buildings | 96.89 | 96.85 | -0.04 |

The released qualitative examples for powerlines, fences, and poles illustrate
local recovery cases only. They must not be interpreted as aggregate
class-level TargetMOCHA improvements.

## Reproducibility

Canonical configurations are provided under `configs/paper/`.

Released checkpoints are indexed in `MODEL_ZOO.md` and hosted through the
corresponding Hugging Face release.

Checkpoint and experiment provenance is retained under `provenance/`.
