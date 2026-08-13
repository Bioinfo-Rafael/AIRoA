#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m airoa.pi05.train --config "${CONFIG:-$ROOT/configs/pi05_track1.yaml}" \
  --output-dir "${OUTPUT_DIR:-$ROOT/outputs/pi05_track1}" \
  --steps "${TRAIN_STEPS:-1000000}" --hours "${TRAIN_HOURS:-7}"
