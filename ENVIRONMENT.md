# MOCHA environments

## Research training and evaluation

The canonical research environment was based on Pointcept/Utonia with:

- Python 3.10
- PyTorch 2.5.0
- CUDA 12.4
- torchvision 0.20.0
- torchaudio 2.5.0
- spconv-cu124
- NumPy <= 1.26.4 for the Utonia environment

The original environment specifications are retained under `environment/`.

## Validated Jetson deployment

The final Jetson validation used the retained `mocha-edge:jp6-cuda-scatter`
container.

The captured device/container state is retained under `environment/`,
including the exact pip lock, system package list, Docker image metadata,
and Jetson system information.

The validated deployment policy uses FP16 checkpoint storage while keeping
the model FP32-resident and using CUDA autocast FP16 during inference.

True FP16 model residency is not the supported deployment path.
