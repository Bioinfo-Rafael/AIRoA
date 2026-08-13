#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="full"
FRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$MODE" == "smoke" ]]; then
  CONFIG="$ROOT/configs/smoke.yaml"
  OUTPUT="$ROOT/outputs/pi05_track1_smoke"
  STEPS="${TRAIN_STEPS:-1}"
  HOURS="${TRAIN_HOURS:-0}"
else
  CONFIG="$ROOT/configs/pi05_track1.yaml"
  OUTPUT="$ROOT/outputs/pi05_track1"
  STEPS="${TRAIN_STEPS:-1000000}"
  HOURS="${TRAIN_HOURS:-7}"
fi
mkdir -p /tmp/airoa-home "$ROOT/artifacts"
nvidia-smi
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
if [[ "$FRESH" == 1 && -d "$OUTPUT" ]]; then
  ARCHIVE="$ROOT/outputs/archive/$(basename "$OUTPUT")-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$(dirname "$ARCHIVE")"
  mv "$OUTPUT" "$ARCHIVE"
fi
python "$ROOT/scripts/environment_check.py"
bash "$ROOT/scripts/setup_env.sh"
python "$ROOT/scripts/inspect_dataset.py" --output "$ROOT/artifacts/dataset_metadata.json"
ARGS=(--config "$CONFIG" --output-dir "$OUTPUT" --steps "$STEPS" --hours "$HOURS")
python -m airoa.pi05.train "${ARGS[@]}"
python -m airoa.pi05.evaluate --config "$CONFIG" --output-dir "$OUTPUT" --artifacts-dir "$ROOT/artifacts"
cp "$OUTPUT/training_summary.json" "$ROOT/artifacts/training_summary.json"
python "$ROOT/scripts/build_submission.py" --selected-model "$ROOT/artifacts/selected_model.json" --output "$ROOT/artifacts/submission.zip"
bash "$ROOT/scripts/validate_submission.sh" "$ROOT/artifacts/submission.zip"
printf 'pipeline_status=PASS\nmode=%s\ncheckpoint=%s\nsubmission=%s\n' \
  "$MODE" "$(python -c 'import json; print(json.load(open("artifacts/selected_model.json"))["checkpoint"])')" \
  "$ROOT/artifacts/submission.zip" > "$ROOT/artifacts/final_report.txt"
