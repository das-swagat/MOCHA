# Model Weights License

## Scope

This file governs the model checkpoints and checkpoint-bearing deployment
artifacts released as part of MOCHA, including the corresponding artifacts
distributed through:

https://huggingface.co/das-swagat/MOCHA

It does not govern source code or third-party datasets.

## License

Unless explicitly stated otherwise, MOCHA model weights are made available
under the Creative Commons Attribution-NonCommercial 4.0 International
license (CC BY-NC 4.0):

https://creativecommons.org/licenses/by-nc/4.0/

The MOCHA weights are freely available for use, sharing, and adaptation for
non-commercial purposes under CC BY-NC 4.0, subject to its attribution and
other applicable terms.

Commercial use is not granted by this release.

## MOCHA and Utonia provenance

The canonical MOCHA pretraining lineage was initialized from the official
Utonia pretraining checkpoint:

`pretrain-utonia-v1m1-0-base_stagev2.pth`

SHA256:

`9f38909449a3796295f5fcd60940a72bba870136e0577642f270a3ed668666bd`

The official Utonia model repository distributes this checkpoint under
CC BY-NC 4.0.

Accordingly, MOCHA checkpoints descending from this initialization retain
the applicable Utonia attribution and non-commercial restriction.

Utonia:
https://huggingface.co/Pointcept/Utonia

## MultiMOCHA and TargetMOCHA provenance

The canonical MultiMOCHA-15k pretraining run was initialized from scratch
(`weight = None`) and does not descend from the Utonia pretrained weights.

MultiMOCHA-15k and the released MultiMOCHA/TargetMOCHA downstream weights
are nevertheless released under CC BY-NC 4.0 as the MOCHA project's
weight-distribution policy.

Canonical checkpoint ancestry and SHA256 provenance are maintained under
`provenance/`.

## Dataset terms

No training or evaluation datasets are redistributed with these weights.

The datasets used by MOCHA have their own licenses, terms of use, and access
conditions. This weight license does not grant rights to any dataset and does
not override any applicable third-party terms.

See `DATASETS.md`.

Users are responsible for determining whether their intended use complies
with the terms applicable to the datasets, software, and other third-party
resources associated with a particular model.

## Edge and optimized weights

Quantized, precision-converted, compact, inference-ready, or Jetson-oriented
versions of a checkpoint remain subject to the same weight license and
applicable upstream terms as their parent checkpoint.

Source code bundled with an edge package is governed separately by the
repository software license and `THIRD_PARTY_NOTICES.md`.

## Attribution

When using or redistributing these weights, retain this license notice and
provide appropriate attribution to the MOCHA work.

If you use these weights in research, please cite the MOCHA paper using the
citation information in `CITATION.cff` and link to the official MOCHA repository.

For checkpoints derived from the Utonia initialization, also provide
appropriate attribution to Utonia/Pointcept.

Final paper citation metadata is provided in `CITATION.cff`.

## No additional dataset rights

Downloading a MOCHA checkpoint does not provide, imply, or transfer access
rights to any dataset used for training or evaluation.
