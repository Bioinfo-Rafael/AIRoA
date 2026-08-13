#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AIROA_TRAIN_IMAGE:-airoa-pi05-train:0.1.0-cu130}"
bash "$ROOT/scripts/docker_build.sh"
mkdir -p "$ROOT/artifacts/environment"
if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  docker run --rm --gpus all \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -v "$ROOT:/workspace" \
    -v "$ROOT/cache/huggingface:/cache/huggingface" \
    -v "$ROOT/data:/data" \
    -w /workspace "$IMAGE" \
    bash -lc 'nvidia-smi && python scripts/environment_check.py && python -c "import lerobot, transformers; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print(PI05Policy.__name__)"'
else
  echo "NVIDIA GPU passthrough unavailable on this host; running pinned container import test only." >&2
  docker run --rm \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -v "$ROOT:/workspace" \
    -w /workspace "$IMAGE" \
    bash -lc 'python scripts/environment_check.py --allow-no-gpu && python -c "import lerobot, transformers; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print(PI05Policy.__name__)"'
fi
