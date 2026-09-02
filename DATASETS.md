# Datasets

MOCHA does not redistribute, mirror, sublicense, or provide copies of any
third-party dataset.

Users must obtain each dataset independently from its original provider and
comply with the provider's current license, terms of use, access conditions,
and citation requirements.

## Datasets used in this work

| Dataset | Official source | Usage note |
|---|---|---|
| nuScenes | https://www.nuscenes.org/nuscenes | Used for MOCHA pretraining/evaluation and in MultiMOCHA. Subject to the nuScenes Terms of Use. |
| SemanticKITTI | https://semantic-kitti.org/ | Used in MultiMOCHA and downstream evaluation. SemanticKITTI annotations and underlying KITTI data are governed by their respective provider terms. |
| Toronto-3D | https://github.com/WeikaiTan/Toronto-3D | Used in MultiMOCHA and downstream evaluation. Provider states CC BY-NC 4.0 for the dataset. |
| RELLIS-3D | https://github.com/unmannedlab/RELLIS-3D | Used in MultiMOCHA and downstream evaluation. Provider states CC BY-NC-SA 3.0. |
| SemanticUSL | https://www.unmannedlab.org/research/SemanticUSL | Used in MultiMOCHA and downstream evaluation. No explicit dataset license was identified on the official provider page; users are responsible for verifying current provider terms before use. |
| DALES | https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales | Used only for downstream aerial evaluation/adaptation and was excluded from MultiMOCHA pretraining. Provider terms restrict commercial use. |

## Important

The source-code license for this repository does **not** grant any rights to the
datasets listed above.

Likewise, availability of MOCHA model checkpoints does not grant access to,
or replace the terms governing, the datasets on which models were trained or
evaluated.

Dataset providers may update their terms. The original provider's current
terms always take precedence over this summary.

Users should cite the original dataset publications when using the
corresponding datasets.

## Additional LiDAR resources

For a broader provenance-tracked catalog of LiDAR datasets, methods,
benchmarks, official access routes, and licensing information, see:

https://github.com/das-swagat/lidar-research-atlas

The LiDAR Research Atlas is an informational index and does not redistribute
the datasets listed above.
