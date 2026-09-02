#!/usr/bin/env bash
set -euo pipefail

IMAGE="mocha-edge:jp6"

echo "Building $IMAGE"
echo "This is the one potentially slow step: spconv is compiled for Orin SM 8.7."

sudo docker build \
  --network=host \
  -t "$IMAGE" \
  -f Dockerfile .

echo
echo "===== VERIFY IMAGE ====="
sudo docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  "$IMAGE" \
  python3 -c 'import torch,spconv,torch_scatter,addict,timm; print("torch",torch.__version__); print("cuda",torch.version.cuda); print("gpu",torch.cuda.get_device_name(0)); print("spconv",spconv.__version__); print("torch_scatter",torch_scatter.__version__); print("DEPENDENCIES: PASS")'
