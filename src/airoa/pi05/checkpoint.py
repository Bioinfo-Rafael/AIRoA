"""Pinned checkpoint download and official LeRobot π0.5 loading."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from airoa.constants import (
    MODEL_ID,
    MODEL_REVISION,
    RENAME_MAP,
    TOKENIZER_SHA256,
    TOKENIZER_URL,
)

TOKENIZER_PATTERNS = ["tokenizer.model"]


def download_checkpoint(cache_dir: str | Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def download_tokenizer(cache_dir: str | Path | None = None) -> Path:
    """Cache OpenPI's anonymous, immutable PaliGemma SentencePiece model.

    The HF PaliGemma repository is gated. OpenPI uses this same public GCS
    object with anonymous credentials, so it is the reproducible source for
    training and for the offline submission.
    """
    root = Path(cache_dir or os.environ.get("HF_HOME", ".cache/huggingface"))
    destination = root / "openpi" / TOKENIZER_SHA256 / "tokenizer.model"
    if destination.is_file() and sha256_file(destination) == TOKENIZER_SHA256:
        return destination.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    urllib.request.urlretrieve(TOKENIZER_URL, temporary)
    found = sha256_file(temporary)
    if found != TOKENIZER_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"PaliGemma tokenizer checksum mismatch: {found}")
    temporary.replace(destination)
    return destination.parent


class OpenPIPaligemmaTokenizer:
    """Minimal HF-call-compatible wrapper around OpenPI's SentencePiece tokenizer."""

    def __init__(self, model_file: str | Path) -> None:
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(model_file))
        self.pad_token_id = 0
        self.bos_token_id = int(self.processor.bos_id())
        self.eos_token_id = int(self.processor.eos_id())
        if (self.pad_token_id, self.bos_token_id, self.eos_token_id) != (0, 2, 1):
            raise RuntimeError("Unexpected PaliGemma SentencePiece special-token ids")

    def __call__(
        self,
        texts,
        *,
        max_length: int,
        truncation: bool,
        padding: str,
        padding_side: str = "right",
        return_tensors: str = "pt",
        **_kwargs,
    ):
        import torch

        if isinstance(texts, str):
            texts = [texts]
        if padding != "max_length" or padding_side != "right" or return_tensors != "pt":
            raise ValueError("π0.5 tokenizer requires right max-length PyTorch padding")
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for text in texts:
            ids = list(self.processor.encode(str(text), add_bos=True, add_eos=False))
            if len(ids) > max_length:
                if not truncation:
                    raise ValueError(f"Prompt has {len(ids)} tokens, limit is {max_length}")
                ids = ids[:max_length]
            mask = [1] * len(ids)
            padding_count = max_length - len(ids)
            rows.append(ids + [self.pad_token_id] * padding_count)
            masks.append(mask + [0] * padding_count)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def assert_checkpoint_contract(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete: {missing}")
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    mapping = config.get("normalization_mapping", {})
    expected = {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}
    if mapping != expected:
        raise ValueError(f"Refusing normalization drift: expected {expected}, found {mapping}")
    if config.get("dtype") != "bfloat16":
        raise ValueError(f"Expected checkpoint bfloat16 dtype, found {config.get('dtype')}")
    if config.get("chunk_size") != 50 or config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        raise ValueError("Checkpoint is not the expected 50x7 LIBERO π0.5 policy")
    return config


def load_policy_and_processors(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    device: str,
    training: bool,
    train_expert_only: bool = True,
    gradient_checkpointing: bool = True,
    compile_model: bool = False,
):
    """Load model and the checkpoint's own normalizer/unnormalizer without replacing stats."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint_dir = Path(checkpoint_dir).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    assert_checkpoint_contract(checkpoint_dir)
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    config.device = device
    config.compile_model = compile_model
    config.gradient_checkpointing = bool(training and gradient_checkpointing)
    config.train_expert_only = bool(training and train_expert_only)
    config.freeze_vision_encoder = bool(training)
    config.push_to_hub = False
    config.repo_id = None
    policy = PI05Policy.from_pretrained(
        checkpoint_dir,
        config=config,
        local_files_only=True,
        strict=True,
    )
    _assert_weight_loaded(policy, checkpoint_dir / "model.safetensors")
    preprocessor, postprocessor = load_processors(checkpoint_dir, tokenizer_dir, device=device)
    if training:
        policy.train()
    else:
        policy.eval()
    return policy, preprocessor, postprocessor


def load_processors(checkpoint_dir: str | Path, tokenizer_dir: str | Path, *, device: str):
    """Load only the serialized processor contract for low-memory tests."""
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    checkpoint_dir = Path(checkpoint_dir).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    assert_checkpoint_contract(checkpoint_dir)
    tokenizer_file = tokenizer_dir / "tokenizer.model"
    if not tokenizer_file.is_file() or sha256_file(tokenizer_file) != TOKENIZER_SHA256:
        raise RuntimeError(f"Missing or unverified OpenPI tokenizer: {tokenizer_file}")
    tokenizer = OpenPIPaligemmaTokenizer(tokenizer_file)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_dir),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
            "tokenizer_processor": {"tokenizer": tokenizer, "tokenizer_name": None},
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_dir),
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _assert_weight_loaded(policy, model_file: Path) -> None:
    """Catch v0.4.4's defensive loader path returning a randomly initialized model."""
    import torch
    from safetensors import safe_open

    model_state = policy.state_dict()
    with safe_open(model_file, framework="pt", device="cpu") as source:
        for source_key in source.keys():
            mapped = source_key
            if mapped.startswith("action_time_mlp_in."):
                mapped = mapped.replace("action_time_mlp_in.", "time_mlp_in.")
            elif mapped.startswith("action_time_mlp_out."):
                mapped = mapped.replace("action_time_mlp_out.", "time_mlp_out.")
            if not mapped.startswith("model."):
                mapped = f"model.{mapped}"
            if mapped not in model_state:
                continue
            expected = source.get_tensor(source_key)
            if expected.numel() > 1_000_000:
                continue
            actual = model_state[mapped].detach().cpu()
            if actual.shape != expected.shape or not torch.equal(actual, expected.to(actual.dtype)):
                raise RuntimeError(f"Checkpoint tensor was not loaded correctly: {source_key} -> {mapped}")
            return
    raise RuntimeError("Could not find a small checkpoint tensor to verify after π0.5 load")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
