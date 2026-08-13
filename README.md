```bash
TRAIN_HOURS=7 bash scripts/docker_run_all.sh --mode full
```

# AIRoA: π0.5 for PARC 2026 Track 1

This repository fine-tunes the official LeRobot v0.4.4 PyTorch π0.5 implementation from
`lerobot/pi05_libero_finetuned_v044` on pinned `lerobot/libero_plus`, selects a recent checkpoint and
execution horizon, exports an offline PARC policy server, creates `artifacts/submission.zip`, and runs the
pinned official validator. Docker is the canonical execution path; no host Python or CUDA toolkit is used.

## Prerequisites

- Linux x86_64
- Docker with BuildKit
- NVIDIA driver compatible with CUDA 13 and NVIDIA Container Toolkit
- NVIDIA GPU with at least 24 GB VRAM (L4/RTX 3090) and sufficient disk (roughly 80 GB free is recommended)
- Git and network access for the first model/dataset/image download

The image and all major dependencies are pinned. Model, dataset, checkpoints, caches, and submission artifacts
live in mounted host directories and are not baked into the image. A rebuild reuses `cache/huggingface`, `data`,
and `outputs`.

Apple Silicon can build the pinned `linux/amd64` image, but it cannot run CUDA training. `docker_run_all.sh`
detects this and exits after the build with a diagnostic instead of silently running a fake CPU training step.

## One-step smoke test

The smoke command follows the same real data/model/training/export code path as the full run. It downloads the
smallest dataset shard containing episode 0, executes a finite forward/loss/backward/optimizer step, checks an
action projection changed, saves/reloads the checkpoint in a separate evaluation process, performs real
50×7 inference, exercises the PARC adapter, builds the zip, and invokes the official validator.

```bash
TRAIN_STEPS=1 bash scripts/docker_run_all.sh --mode smoke
```

Use `--fresh` only when a new run is desired. Otherwise the latest checkpoint and optimizer state are resumed.
Existing runs passed with `--fresh` are moved into `outputs/archive/`; they are not deleted.

## Full run controls

`TRAIN_HOURS=7` is the default stopping window and `TRAIN_STEPS` is a hard upper bound/override. The loop stops
when either bound is reached. Common overrides are passed through Docker:

```bash
BATCH_SIZE=1 NUM_WORKERS=2 SAVE_FREQ=500 TRAIN_HOURS=7 \
  bash scripts/docker_run_all.sh --mode full
```

The primary 24 GB configuration uses bfloat16, `train_expert_only=true`, VLM/vision freezing, gradient
checkpointing, compile disabled, batch size 1, AdamW at `1e-5`, and the checkpoint's own MEAN_STD processor
weights. For 48 GB GPUs, raise `BATCH_SIZE`; GPU names are never hard-coded.

## Model and normalization

- Pretrained model: `lerobot/pi05_libero_finetuned_v044`
- Model revision: `8e174154ef5f6c60a8da12ae99c303d8963138c1`
- LeRobot v0.4.4 revision: `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`
- Patched Transformers revision: `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`
- Dataset revision: `f3f49f426d75030177b18778374005bc12ccd588`

The checkpoint declares MEAN_STD for state/action and IDENTITY for images. Training, evaluation, and submission
load `policy_preprocessor.json`, normalizer weights, `policy_postprocessor.json`, and unnormalizer weights from
the selected checkpoint. The code refuses a normalization contract change.

## Dataset and Track 1 strategy

`scripts/inspect_dataset.py` downloads metadata first and records schema, fps, camera keys, episode/task counts,
and perturbation fields in `artifacts/dataset_metadata.json`. The pinned dataset is LeRobot v3.0 with front and
wrist 256×256 video, 8D state, 7D actions, 20 fps, 40 tasks, 14,347 episodes, and 2,238,036 frames.

The episode metadata does not expose a reliable appearance-perturbation label. The pipeline therefore does not
guess labels from episode ordering; it uses the complete dataset and samples the 40 primitives uniformly.
Both cameras pass through a 128×128 information bottleneck and are restored to checkpoint feature resolution
before the official processor. Mild brightness/contrast/saturation/hue/gamma parameters are sampled once per
observation and shared across front/wrist views. Geometry augmentation and strong texture overlays are off.

## PARC inference

The official template is preserved byte-for-byte outside `MyPolicy`. The adapter:

- rotates agent and wrist images by 180° as in OpenPI LIBERO inference;
- forms 8D state from EEF xyz, robosuite/OpenPI quaternion-to-axis-angle, and two gripper positions;
- lets the official π0.5 model resize 128×128 inputs to 224×224 and normalize exactly once;
- predicts a 50-step chunk, but queues only 5/8/10 steps (default/selected horizon is normally 5);
- clears action queue, previous chunk, instruction state, and deterministic noise generator on reset.

The generated server is offline: weights, tokenizer assets, LeRobot, the exact patched Transformers source,
and required binary bindings are inside the zip. It does not put torch in `requirements.txt`, so PARC's
preinstalled torch 2.11.0+cu130 remains untouched.

The Hugging Face PaliGemma tokenizer repository is gated. To avoid a hidden token requirement, the pipeline
uses the identical anonymous SentencePiece object referenced by official OpenPI and verifies its pinned SHA-256
before use/export.

## Evaluation and artifacts

The small checkpoint gate evaluates up to the last three checkpoints and horizons 5/8/10 on a fixed real
LIBERO-Plus batch. It records action MSE, latency, CUDA memory, action norm, xyz/rotation jerk, action cosine,
gripper flips, trajectory/rotation length, and SPARC proxy. `success` and `collision` are explicitly null for
this offline gate; they are not fabricated. Actual success/collision require a simulator rollout.

The pipeline creates:

```text
artifacts/
├── submission.zip
├── evaluation_summary.csv
├── evaluation_summary.md
├── inference_benchmark.json
├── training_summary.json
├── selected_model.json
├── submission_build.json
└── environment/
    ├── docker_image_id.txt
    ├── docker_inspect.json
    ├── pip_freeze.txt
    ├── python_version.txt
    ├── torch_version.txt
    ├── cuda_version.txt
    ├── gpu_info.txt
    ├── git_commit.txt
    └── dataset_revision.txt
```

## Validation

`scripts/validate_submission.sh` pins `matsuolab/PARC2026_pre` at
`eb8a063cf1d69615754b0cbb31b0a9162621ec9b`, runs static validation and then `/health`, `/reset`, `/act` dynamic
validation. Real π0.5 is used when CUDA is visible; a deterministic stub is used only for HTTP compatibility on
a non-GPU host and is reported as such.

To additionally build the unmodified official CPU validator Docker image and run its static check:

```bash
OFFICIAL_DOCKER=1 TRAIN_HOURS=7 bash scripts/docker_run_all.sh --mode full
```

The official CPU image cannot numerically run this CUDA π0.5 policy. Its role here is zip/requirements/template
compatibility; real inference is tested in the training GPU container and PARC's CUDA environment.

## Repository layout

```text
configs/                 smoke and primary experiment settings
docker/                  exact dependency locks and constraints
scripts/                 Docker, training, evaluation, export, and validation entry points
src/airoa/data/          LIBERO-Plus metadata, transforms, and balanced sampler
src/airoa/pi05/          official checkpoint integration, training, and evaluation
src/airoa/parc/          observation adapter and offline chunked runtime
src/airoa/metrics/       trajectory/smoothness proxies
submission/              official PARC template with only MyPolicy edited
tests/                   adapter, augmentation, metrics, queue/reset tests
docs/                    design, environment, and debug decisions
```

## Troubleshooting

- GPU unavailable: confirm Linux, `nvidia-smi`, `docker info`, NVIDIA Container Toolkit, and `--gpus all`.
- OOM: keep batch size 1, workers 0–2, gradient checkpointing on, compile off, and expert-only training on.
- torch conflict: do not install LeRobot with dependency resolution; the image intentionally installs torch
  2.11+cu130 first and installs LeRobot v0.4.4 with `--no-deps`.
- Video decode failure: primary backend is torchcodec 0.11.1+cu130 with ffmpeg present; set `video_backend: pyav`
  only as a fallback.
- Interrupted overnight run: rerun the same command. Checkpoints and optimizer state are mounted on the host.
- Offline submission failure: confirm the selected model directory includes both processor state files and the
  local `tokenizer/` directory. `submission_manifest.json` records hashes and exact revisions.

See [environment details](docs/ENVIRONMENT.md), [design rationale](docs/DESIGN.md), the
[performed verification](docs/VERIFICATION.md), and the [debug log](docs/ENV_DEBUG_LOG.md).
