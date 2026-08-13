# Design

## Training boundary

AIRoA does not reimplement π0.5. `PI05Policy`, its loss, flow-matching sampler, resize-with-pad, tokenizer state
prompt, and checkpoint serialization come from the pinned LeRobot v0.4.4 source. AIRoA owns only orchestration,
LIBERO-Plus adaptation/augmentation, checkpoint selection, PARC observation adaptation, and packaging.

The checkpoint is loaded after setting `train_expert_only`, vision freeze, bfloat16, gradient checkpointing, and
compile flags so `requires_grad` is correct at construction time. An invariant verifies every PaliGemma VLM
parameter is frozen. The optimizer receives only trainable parameters; action expert and projection parameters
remain trainable.

## Processor contract

The original preprocessor and postprocessor JSON/safetensor state are loaded for every phase. Overrides are
limited to the checksum-verified OpenPI SentencePiece tokenizer object, device, and dataset camera-key
renaming. Normalizer statistics are never
replaced with LIBERO-Plus statistics. This is deliberate because the selected public checkpoint uses MEAN_STD,
whereas the generic v0.4.4 π0.5 dataclass defaults to QUANTILES.

## Data

The dataset is pinned at a LeRobot v3.0 revision. Smoke uses episode 0, which causes LeRobot to fetch one data
file and one front/wrist video shard rather than the entire corpus. Full training uses all episodes. Since the
metadata has no reliable perturbation column, the sampler balances task primitives only.

The 128 bottleneck uses antialiased bilinear downsampling. Restoring the original feature resolution avoids
feature-contract surprises while retaining only 128×128 information; the official model then performs its own
224 resize/pad and [-1,1] conversion.

## Inference and selection

Fixed diagnostic inference constructs the 50×32 flow noise tensor from a seeded CUDA generator. This makes
checkpoint comparisons repeatable without claiming the flow policy is universally deterministic. The gate uses
real held-out observations and action targets, plus smoothness and latency proxies. It deliberately leaves
success/collision null because those require simulator rollouts.

## Submission

The policy template is checked against the pinned official source before/after `MyPolicy`. The export contains
the selected complete checkpoint, tokenizer assets, AIRoA runtime, LeRobot, patched Transformers, tokenizers,
safetensors, sentencepiece, and their dist-info. Torch is excluded and inherited from PARC's CUDA 13 image.
