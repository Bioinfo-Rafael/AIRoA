# Verification performed in the implementation environment

Date: 2026-08-14 (Asia/Tokyo)

The available host was macOS arm64 (Apple M4) with Docker Desktop and no NVIDIA device. Accordingly, CUDA
training and real π0.5 numerical inference were not claimed. The pipeline intentionally rejects those operations
instead of substituting fake data or CPU training.

## Passed

- Built the pinned `linux/amd64` CUDA 13 training image under emulation.
- Imported torch 2.11.0+cu130, torchvision 0.26.0+cu130, TorchCodec 0.11.1+cu130, the pinned patched
  Transformers, LeRobot v0.4.4, `PI05Policy`, `LeRobotDataset`, and processor pipeline.
- Read pinned LIBERO-Plus metadata and decoded real episode 0 front/wrist video plus 8D state and a 50×7
  action target with TorchCodec.
- Loaded the public checkpoint's preprocessor, MEAN_STD state, postprocessor, and OpenPI tokenizer; both a real
  dataset item and a synthetic PARC observation were finite and contract-correct.
- Opened the complete 7,473,096,344-byte public safetensors file (812 tensors) and verified its model revision.
- Passed all unit tests, lint, Python compilation, shell syntax, and byte-identical official-template boundary.
- Exported the real weights and offline runtime as `artifacts/submission.zip` (7.244 GiB) and passed the pinned
  official validator static and dynamic checks with zero errors/warnings. Dynamic HTTP used the explicitly
  selected deterministic stub because CUDA was unavailable; it tested health/reset/three acts/reset/act.
- Built the pinned official PARC Dockerfile unchanged for `linux/amd64` (including LIBERO-Plus registration and
  583 textures) and passed its in-container static validator with zero errors/warnings.

## Hardware-blocked

- A real finite loss, backward pass, optimizer step, changed parameter, fine-tuned checkpoint save/reload.
- Real π0.5 action sampling, L4 memory use, and inference latency.
- GPU rollout success/collision and PARC score.

Run the README's smoke command first on the target Linux NVIDIA host; it traverses the same code path as full
training and makes all of the hardware-blocked checks mandatory.
