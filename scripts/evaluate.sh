#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m airoa.pi05.evaluate --config "${CONFIG:-$ROOT/configs/pi05_track1.yaml}" \
  --output-dir "${OUTPUT_DIR:-$ROOT/outputs/pi05_track1}" --artifacts-dir "$ROOT/artifacts"
