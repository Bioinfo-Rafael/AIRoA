#!/usr/bin/env bash
set -euo pipefail

# Never send local policy-server health/reset/act requests through the host HTTP proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT/artifacts/submission.zip}"
bash "$ROOT/scripts/setup_env.sh"
VALIDATOR="$ROOT/external/PARC2026_pre/validate_submission.py"
mkdir -p "$ROOT/artifacts"
python "$VALIDATOR" "$TARGET" --static | tee "$ROOT/artifacts/validator_static.log"
if python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  BACKEND=pi05
else
  BACKEND=stub
  echo "GPU unavailable: dynamic validator uses deterministic stub; real π0.5 startup remains GPU-only." >&2
fi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
AIROA_POLICY_BACKEND="$BACKEND" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python "$VALIDATOR" "$TARGET" | tee "$ROOT/artifacts/validator_dynamic.log"
printf 'validator=PASS\nbackend=%s\nparc_revision=%s\n' "$BACKEND" \
  "$(git -C "$ROOT/external/PARC2026_pre" rev-parse HEAD)" > "$ROOT/artifacts/submission_validation.txt"
