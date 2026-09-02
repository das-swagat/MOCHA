# Installation

MOCHA is implemented as an extension of the Pointcept/PTv3 research stack.
This repository contains the MOCHA-specific source, configurations,
provenance, model utilities, and validated edge-deployment code. It is not
a complete redistribution of Pointcept.

## Lightweight model utilities

Install the lightweight requirements and list released models:

    python3 -m pip install -r requirements.txt
    python3 tools/list_models.py

Weights are hosted on Hugging Face and are not stored in this GitHub repository.

## Research environment

The validated research stack used Python 3.10, PyTorch 2.5.0, CUDA 12.4,
torchvision 0.20.0, torchaudio 2.5.0, and spconv-cu124.

Captured environment specifications are retained in:

- `environment/environment_pointcept.yml`
- `environment/environment_utonia.yml`
- `ENVIRONMENT.md`

These are reproducibility records. The Pointcept environment references
Pointcept-local compiled operators and should be used with a compatible
Pointcept checkout.

MOCHA-specific implementation files are under `src/pointcept/`.
Paper configurations are under `configs/`.

## Model weights

List released models:

    python3 tools/list_models.py --details

Download and checksum-verify a model, for example:

    python3 tools/download_weights.py --model semkitti-targetmocha

See `MODEL_ZOO.md` for model IDs and `LICENSE-WEIGHTS.md` for weight terms.

## Jetson edge deployment

Validated deployment packages are provided for nuScenes, SemanticKITTI,
Toronto3D, RELLIS-3D, SemanticUSL, and DALES.

Source-side deployment files are under `edge/`. Checkpoint-bearing deployment
packages are distributed through Hugging Face.

Each package contains its matching runtime, Dockerfile, `build_jetson.sh`,
model specification, and smoke-test entry point.

The validated Jetson deployment policy uses:

- FP16 checkpoint storage
- FP32-resident model parameters
- CUDA FP16 autocast during inference
- no test-time augmentation

True FP16 model residency is not claimed as the validated deployment path.

See the matching directory under `edge/` and `ENVIRONMENT.md` for the
captured deployment environment.
