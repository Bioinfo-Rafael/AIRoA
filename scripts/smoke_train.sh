#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_STEPS="${TRAIN_STEPS:-1}" bash "$ROOT/scripts/docker_run_all.sh" --mode smoke "$@"
