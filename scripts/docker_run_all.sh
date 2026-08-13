#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AIROA_TRAIN_IMAGE:-airoa-pi05-train:0.1.0-cu130}"
bash "$ROOT/scripts/docker_build.sh"
if [[ "$(uname -s)" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: full/smoke π0.5 training requires a Linux NVIDIA host with nvidia-smi." >&2
  echo "Docker image is built; move this repository to a Linux GPU host and rerun the same command." >&2
  exit 3
fi
mkdir -p "$ROOT/cache/huggingface" "$ROOT/data" "$ROOT/outputs" "$ROOT/artifacts"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/airoa-home \
  -e HF_HOME=/cache/huggingface \
  -e HF_LEROBOT_HOME=/data/lerobot \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e AIROA_DOCKER_IMAGE_ID="$IMAGE_ID" \
  -e TRAIN_STEPS="${TRAIN_STEPS:-}" \
  -e TRAIN_HOURS="${TRAIN_HOURS:-}" \
  -e BATCH_SIZE="${BATCH_SIZE:-}" \
  -e NUM_WORKERS="${NUM_WORKERS:-}" \
  -e SAVE_FREQ="${SAVE_FREQ:-}" \
  -v "$ROOT:/workspace" \
  -v "$ROOT/cache/huggingface:/cache/huggingface" \
  -v "$ROOT/data:/data" \
  -v "$ROOT/outputs:/workspace/outputs" \
  -v "$ROOT/artifacts:/workspace/artifacts" \
  -w /workspace "$IMAGE" \
  bash scripts/run_all.sh "$@"
if [[ "${OFFICIAL_DOCKER:-0}" == "1" ]]; then
  bash "$ROOT/scripts/validate_official_docker.sh" "$ROOT/artifacts/submission.zip"
fi
