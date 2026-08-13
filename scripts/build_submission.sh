#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/scripts/build_submission.py" --selected-model "$ROOT/artifacts/selected_model.json" \
  --output "$ROOT/artifacts/submission.zip"
