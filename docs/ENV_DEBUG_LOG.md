# Environment debug log

## Host has no NVIDIA GPU

- Symptom: `nvidia-smi` is unavailable; host is Darwin arm64 with Apple M4 and Python 3.14.
- Cause: current machine is not a Linux NVIDIA training host.
- Attempt: verified Docker Desktop, CUDA image multi-arch manifest, disk space, and GitHub authentication.
- Result: Linux/amd64 training image can be built under emulation, but CUDA passthrough and a real training step
  cannot be manufactured on this host.
- Final fix: canonical scripts fail before training when GPU passthrough is absent and emit a precise migration
  instruction. CPU/import and official static/HTTP compatibility tests remain separate from GPU claims.

## LeRobot torch resolver conflict

- Symptom: v0.4.4 `pyproject.toml` specifies `torch>=2.2.1,<2.11.0` and torchvision `<0.26` while PARC pins
  torch 2.11.0+cu130.
- Cause: LeRobot dependency metadata was not yet widened for the PARC torch release.
- Attempt: inspected the tagged source and π0.5 implementation rather than downgrading torch.
- Result: no direct π0.5 source dependency on a removed torch API was found.
- Final fix: install torch/torchvision/torchcodec cu130 first, build exact LeRobot source, and install its wheel
  with `--no-deps`. Focused imports and actual GPU smoke are mandatory gates.

## Transformers special branch

- Symptom: LeRobot's π0.5 extra points at a moving `fix/lerobot_openpi` Git branch.
- Cause: the port needs patched SigLIP/PaliGemma behavior and a `transformers.models.siglip.check` guard.
- Attempt: resolved the branch head and inspected its reported package version/guard.
- Result: SHA `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`, version 4.53.3, explicitly accepts 4.53.2/4.53.3.
- Final fix: fetch/build by immutable SHA and vendor that installed build into the submission.

## LeRobot eager optional-policy imports

- Symptom: `from lerobot.policies.pi05...` failed on missing `diffusers` while importing unrelated GR00T code.
- Cause: v0.4.4's parent `lerobot.policies` initializer eagerly registers every policy, and some child package
  initializers eagerly import optional model implementations.
- Attempt: traced the import chain through `groot/__init__.py`; adding GR00T dependencies would also pull PEFT and
  other hardware/model packages that π0.5 never calls.
- Result: the failure is an import-boundary bug, not a π0.5 dependency.
- Final fix: after installing the exact upstream wheel, replace only the parent package initializer with an empty
  documented boundary and the runtime-unused `TrainPipelineConfig` annotation import with `object`. This also
  prevents unrelated env→robot→serial imports. The teleoperator parent initializer is similarly narrowed because
  the processor only imports its event enum. π0.5/config/processor/model files remain byte-identical and are
  imported explicitly.

## LIBERO-Plus perturbation labels

- Symptom: requested Light Conditions/Background Texture/table balancing cannot be derived from task or episode
  metadata fields.
- Cause: pinned v3.0 episode metadata contains task, video ranges, lengths, and stats but no perturbation column.
- Attempt: scanned every episode schema field for perturb/light/texture/background/table tokens.
- Result: no reliable column exists; episode ordering is undocumented.
- Final fix: do not guess. Use the full LIBERO-Plus corpus and uniform 40-task sampling.

## Video backend

- Symptom: TorchCodec 0.10+cu130 failed to import with torch 2.11 (`undefined symbol` from libtorchcodec).
- Cause: the official compatibility matrix maps TorchCodec 0.10 to torch 2.10 and 0.11 to torch 2.11;
  LeRobot v0.4.4's `<0.11` metadata predates torch 2.11.
- Attempt: verified FFmpeg 4 shared libraries were present and inspected every attempted loader error before
  changing versions.
- Result: missing FFmpeg 5–8 errors were expected; the FFmpeg 4 error proved the torch ABI mismatch.
- Final fix: explicitly override to torchcodec 0.11.1+cu130 after the no-deps LeRobot installation. Keep PyAV
  configurable as a fallback and require a real episode-0 decode in smoke.

## Gated PaliGemma tokenizer

- Symptom: anonymous requests to the processor's `google/paligemma-3b-pt-224` tokenizer return HTTP 401.
- Cause: the Google Hugging Face repository requires license acceptance even though the LeRobot checkpoint and
  LIBERO-Plus data are public.
- Attempt: inspected the official OpenPI tokenizer path and the gated file metadata (4,264,023 bytes).
- Result: OpenPI anonymously downloads the corresponding SentencePiece model from the public Big Vision bucket.
- Final fix: download that exact OpenPI object, require SHA-256
  `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`, and use a small wrapper that matches
  PaliGemma's BOS/right-padding behavior. The verified file is copied into the offline submission.

## Smoothness unit-test precision

- Symptom: the constant-action cosine test produced `0.9999998807907104` instead of bit-exact `1.0`.
- Cause: float32 norm/division rounding; the metric itself was finite and correct.
- Attempt: verified zero jerk and identical input vectors.
- Result: only an exact-equality test assertion was invalid.
- Final fix: retain float32 production behavior and compare cosine with a `1e-6` absolute tolerance.

## Checkpoint download memory

- Symptom: the 7.47 GB public checkpoint download exited 137 in an 8 GB Docker Desktop VM.
- Cause: Xet high-performance range downloads buffered roughly 5.8 GB while reconstructing the large blob.
- Attempt: monitored container memory, network bytes, incomplete blob size, and Xet range-request logs.
- Result: network transfer was progressing; neither disk space nor model code caused the termination.
- Final fix: default `HF_HUB_DISABLE_XET=1` so Hub uses a streaming HTTP download into the mounted cache.
  Advanced users may explicitly set it to `0` on higher-memory hosts.

## Hugging Face cache root semantics

- Symptom: an offline processor re-save test could not find an already complete model snapshot.
- Cause: `huggingface_hub` interprets an explicit `cache_dir` as the Hub cache directory itself, while `HF_HOME`
  is its parent and normally stores snapshots below `HF_HOME/hub`.
- Attempt: compared the resolved snapshot path with the path generated by passing `/cache/huggingface` directly.
- Result: the explicit argument searched a parallel `models--*` tree and would have downloaded a duplicate model.
- Final fix: model download now lets `huggingface_hub` resolve the pinned snapshot from `HF_HOME`; the custom
  OpenPI tokenizer continues to use `HF_HOME/openpi` intentionally.

## Submission vendor import mapping

- Symptom: the first full 7.47 GB checkpoint export stopped before zip creation because `yaml_include` was not
  importable.
- Cause: the distribution is named `pyyaml-include`, but version 1.4.1 exposes the module as `yamlinclude`.
- Attempt: inspected the installed wheel file list and `top_level.txt` rather than dropping the dependency.
- Result: the model file had copied correctly; only the strict vendor import mapping was wrong.
- Final fix: map the exact distribution to `yamlinclude`, rebuild the clean staging tree, and rerun the offline
  origin check for every runtime dependency.

## Vendored wheel native libraries

- Symptom: the strict offline runtime process found `PIL._imaging` but failed to load its hashed `libtiff`.
- Cause: Pillow stores wheel-owned shared libraries in `pillow.libs`, named after the distribution, not the
  import package `PIL`.
- Attempt: enumerated every `.libs` directory in the exact x86_64 training image.
- Result: the same pattern matters for NumPy, torchvision, and other binary wheels.
- Final fix: copy both import-name and PEP 503-normalized distribution-name `.libs` directories, then require a
  clean-process processor load before creating the zip.

## Validator junk warning

- Symptom: the first weight-bearing zip passed static and dynamic validation but reported `__pycache__` junk.
- Cause: the clean-process offline import gate ran against the staging tree and Python wrote fresh bytecode there
  after the package-copy ignore filters had already run.
- Attempt: located every warned zip entry and confirmed none came from the source package copy itself.
- Result: the warning was harmless but avoidable and added thousands of entries.
- Final fix: disable bytecode writes during the gate and remove any cache/bytecode defensively before zip/CRC.

## Official validator Docker architecture

- Symptom: Docker Desktop initially resolved the official Ubuntu base to arm64 on the Apple Silicon host.
- Cause: the official Dockerfile does not declare a target platform, so Docker follows the host architecture.
- Attempt: stopped that build before dependency installation and compared it with PARC's documented x86_64
  scoring environment.
- Result: arm64 static validation would not prove x86_64 binary-package compatibility.
- Final fix: keep the official Dockerfile unmodified but pass `--platform linux/amd64` to build and run; retain
  `PARC_DOCKER_PLATFORM` only as an explicit diagnostic override.
