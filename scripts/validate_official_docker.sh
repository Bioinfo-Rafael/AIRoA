#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT/artifacts/submission.zip}"
TARGET="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
test -f "$TARGET"
bash "$ROOT/scripts/setup_env.sh"
IMAGE="parc2026-validator:eb8a063c"
PLATFORM="${PARC_DOCKER_PLATFORM:-linux/amd64}"
if [[ "${PARC_DOCKER_REBUILD:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build --platform "$PLATFORM" -t "$IMAGE" "$ROOT/external/PARC2026_pre"
fi
docker run --rm --platform "$PLATFORM" -v "$TARGET:/sub.zip:ro" "$IMAGE" \
  python validate_submission.py /sub.zip --static | tee "$ROOT/artifacts/validator_official_docker.log"
