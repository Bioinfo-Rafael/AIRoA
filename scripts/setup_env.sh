#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARC_SHA="eb8a063cf1d69615754b0cbb31b0a9162621ec9b"
mkdir -p "$ROOT/cache/huggingface" "$ROOT/data" "$ROOT/outputs" "$ROOT/artifacts/environment" "$ROOT/external"
if [[ ! -d "$ROOT/external/PARC2026_pre/.git" ]]; then
  git init "$ROOT/external/PARC2026_pre"
  git -C "$ROOT/external/PARC2026_pre" remote add origin https://github.com/matsuolab/PARC2026_pre.git
fi
if [[ "$(git -C "$ROOT/external/PARC2026_pre" rev-parse HEAD 2>/dev/null || true)" != "$PARC_SHA" ]]; then
  git -C "$ROOT/external/PARC2026_pre" fetch --depth 1 origin "$PARC_SHA"
  git -C "$ROOT/external/PARC2026_pre" checkout --detach FETCH_HEAD
fi
test "$(git -C "$ROOT/external/PARC2026_pre" rev-parse HEAD)" = "$PARC_SHA"
printf '%s\n' "$PARC_SHA" > "$ROOT/artifacts/environment/parc_revision.txt"
printf '%s\n' "f3f49f426d75030177b18778374005bc12ccd588" > "$ROOT/artifacts/environment/dataset_revision.txt"
