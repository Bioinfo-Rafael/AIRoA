#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AIROA_TRAIN_IMAGE:-airoa-pi05-train:0.1.0-cu130}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
mkdir -p "$ROOT/artifacts/environment"
docker build --platform "$PLATFORM" --progress plain -f "$ROOT/Dockerfile.train" -t "$IMAGE" "$ROOT"
docker image inspect "$IMAGE" > "$ROOT/artifacts/environment/docker_inspect.json"
docker image inspect --format '{{.Id}}' "$IMAGE" > "$ROOT/artifacts/environment/docker_image_id.txt"
printf 'image=%s\nplatform=%s\n' "$IMAGE" "$PLATFORM"
