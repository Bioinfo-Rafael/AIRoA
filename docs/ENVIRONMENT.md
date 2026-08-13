# Environment

## Canonical image

- Platform: Linux x86_64
- Base: `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04`
- Base amd64 digest: `sha256:6b285b08f9cf4e466fa010f333cb07bc166fc9d4e59554603fb25504ea0807d2`
- Python: Ubuntu system Python 3.10
- torch: `2.11.0+cu130`
- torchvision: `0.26.0+cu130`
- torchcodec: `0.11.1+cu130` (explicit torch 2.11 ABI override over LeRobot's stale `<0.11` metadata)
- LeRobot: v0.4.4 source SHA `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`
- Transformers: custom π0.5 branch SHA `dcddb970176382c0fcf4521b0c0e6fc15894dfe0` (reports 4.53.3)

The LeRobot and Transformers repositories are fetched by SHA at image build time, made into wheels, and installed
with `--no-deps`. Large data is always mounted. `scripts/environment_check.py` records the actual image runtime,
pip freeze, torch/CUDA, GPU, platform, and git state per run.

## Compatibility

LeRobot v0.4.4 declares torch `<2.11` and torchvision `<0.26`; PARC provides torch 2.11+cu130. The version ceiling
is dependency metadata, not a proven source incompatibility. The image deliberately bypasses that resolver edge
and runs focused import, π0.5 class, patched SigLIP, dataset decode, training, reload, and inference tests.

The submission uses PARC's system torch and pure/source vendoring for LeRobot/Transformers. Linux x86_64 binary
packages for tokenizers, safetensors, and sentencepiece are included. The zip is therefore intended for PARC's
x86_64 GPU evaluator, not macOS or ARM.
