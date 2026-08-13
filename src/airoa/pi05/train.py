"""Minimal, auditable π0.5 fine-tuning loop using official LeRobot model code."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from airoa.config import load_config
from airoa.constants import DATASET_ID, DATASET_REVISION, LEROBOT_REVISION, MODEL_ID, MODEL_REVISION
from airoa.data.libero_plus import (
    PhotometricConfig,
    TaskBalancedSampler,
    make_dataset,
    summarize_sample,
)
from airoa.pi05.checkpoint import (
    download_checkpoint,
    download_tokenizer,
    load_policy_and_processors,
    sha256_file,
)


def _git_sha(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    candidates = sorted(path for path in checkpoint_root.glob("[0-9]*") if path.is_dir())
    return candidates[-1] if candidates else None


def _save_checkpoint(
    output_dir: Path,
    step: int,
    policy,
    preprocessor,
    postprocessor,
    optimizer: torch.optim.Optimizer,
    summary: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"{step:06d}"
    temporary = checkpoint_dir.with_name(f".{checkpoint_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    pretrained_dir = temporary / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    policy.save_pretrained(pretrained_dir)
    preprocessor.save_pretrained(pretrained_dir)
    postprocessor.save_pretrained(pretrained_dir)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()},
        temporary / "training_state.pt",
    )
    (temporary / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    temporary.rename(checkpoint_dir)
    last_link = output_dir / "checkpoints" / "last"
    if last_link.is_symlink() or last_link.exists():
        last_link.unlink()
    last_link.symlink_to(checkpoint_dir.name)
    return checkpoint_dir


def _make_loader(dataset, config: dict[str, Any], seed: int) -> DataLoader:
    balanced = bool(config["dataset"].get("task_balanced", False)) and config["dataset"].get("episodes") is None
    sampler = TaskBalancedSampler(dataset.root, len(dataset), seed=seed) if balanced else None
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train(config_path: Path, output_dir: Path, steps_override: int | None, hours_override: float | None) -> dict:
    config = load_config(config_path)
    if os.environ.get("BATCH_SIZE"):
        config["training"]["batch_size"] = int(os.environ["BATCH_SIZE"])
    if os.environ.get("NUM_WORKERS"):
        config["training"]["num_workers"] = int(os.environ["NUM_WORKERS"])
    if os.environ.get("SAVE_FREQ"):
        config["training"]["save_freq"] = int(os.environ["SAVE_FREQ"])
    seed = int(config.get("seed", 1000))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for real π0.5 fine-tuning. Check `docker run --gpus all`, "
            "NVIDIA Container Toolkit, and the host driver before retrying."
        )
    device = "cuda"
    cache_dir = Path(os.environ.get("HF_HOME", "cache/huggingface"))
    # Let huggingface_hub resolve HF_HOME/hub itself. Passing HF_HOME as
    # cache_dir would create a second, incompatible HF_HOME/models--* tree.
    base_checkpoint = download_checkpoint()
    tokenizer_dir = download_tokenizer(cache_dir)

    dataset_cfg = config["dataset"]
    episodes = dataset_cfg.get("episodes")
    augmentation = PhotometricConfig(**config["augmentation"])
    dataset = make_dataset(
        episodes=episodes,
        video_backend=dataset_cfg.get("video_backend", "torchcodec"),
        bottleneck_size=int(dataset_cfg.get("bottleneck_size", 128)),
        restore_feature_resolution=bool(dataset_cfg.get("restore_feature_resolution", True)),
        augmentation=augmentation,
    )
    real_sample = dataset[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_sample.json").write_text(
        json.dumps(summarize_sample(real_sample), indent=2, default=str) + "\n"
    )
    loader = _make_loader(dataset, config, seed)

    resume_checkpoint = _latest_checkpoint(output_dir) if config["training"].get("resume", True) else None
    model_source = resume_checkpoint / "pretrained_model" if resume_checkpoint else base_checkpoint
    policy, preprocessor, postprocessor = load_policy_and_processors(
        model_source,
        tokenizer_dir,
        device=device,
        training=True,
        train_expert_only=bool(config["model"].get("train_expert_only", True)),
        gradient_checkpointing=bool(config["model"].get("gradient_checkpointing", True)),
        compile_model=bool(config["model"].get("compile_model", False)),
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable π0.5 parameters")
    frozen_vlm = all(
        not parameter.requires_grad
        for parameter in policy.model.paligemma_with_expert.paligemma.parameters()
    )
    if config["model"].get("train_expert_only", True) and not frozen_vlm:
        raise RuntimeError("train_expert_only requested but VLM parameters remain trainable")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"].get("betas", [0.9, 0.95])),
        eps=float(config["training"].get("eps", 1e-8)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    start_step = 0
    if resume_checkpoint:
        state = torch.load(resume_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])

    requested_steps = steps_override if steps_override is not None else int(config["training"]["steps"])
    requested_hours = hours_override if hours_override is not None else float(config["training"].get("hours", 0))
    if start_step >= requested_steps:
        existing_summary = output_dir / "training_summary.json"
        if existing_summary.is_file():
            summary = json.loads(existing_summary.read_text(encoding="utf-8"))
            summary["resume_noop"] = True
            summary["resume_reason"] = f"checkpoint step {start_step} already satisfies target {requested_steps}"
            existing_summary.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
            return summary
        raise RuntimeError(f"Checkpoint step {start_step} already meets target but summary is missing")
    deadline = time.monotonic() + requested_hours * 3600 if requested_hours > 0 else None
    save_freq = max(1, int(config["training"].get("save_freq", requested_steps)))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    tracked_parameter = policy.model.action_out_proj.weight
    before = tracked_parameter.detach().float().cpu().clone()
    data_iterator = iter(loader)
    losses: list[float] = []
    started = time.monotonic()
    step = start_step
    final_checkpoint: Path | None = None
    while step < requested_steps and (deadline is None or time.monotonic() < deadline):
        batch, data_iterator = _next_batch(data_iterator, loader)
        processed = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, details = policy(processed)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}: {grad_norm}")
        optimizer.step()
        step += 1
        losses.append(float(loss.detach().cpu()))
        elapsed = time.monotonic() - started
        print(json.dumps({"step": step, "loss": losses[-1], "grad_norm": float(grad_norm), "elapsed_s": elapsed}))
        if step % save_freq == 0 or step == requested_steps or (deadline and time.monotonic() >= deadline):
            interim = {
                "completed_steps": step,
                "last_loss": losses[-1],
                "finite_loss": True,
                "elapsed_s": elapsed,
            }
            final_checkpoint = _save_checkpoint(
                output_dir, step, policy, preprocessor, postprocessor, optimizer, interim
            )

    if final_checkpoint is None:
        if step == start_step:
            raise RuntimeError("Training deadline elapsed before one optimizer step")
        final_checkpoint = _save_checkpoint(
            output_dir, step, policy, preprocessor, postprocessor, optimizer, {"completed_steps": step}
        )
    delta = float((tracked_parameter.detach().float().cpu() - before).abs().max())
    if delta == 0:
        raise RuntimeError("Optimizer step completed but tracked action projection did not change")
    elapsed = time.monotonic() - started
    summary = {
        "status": "pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(Path(__file__).resolve().parents[3]),
        "docker_image_id": os.environ.get("AIROA_DOCKER_IMAGE_ID"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "lerobot_revision": LEROBOT_REVISION,
        "seed": seed,
        "config": config,
        "normalization": "checkpoint MEAN_STD",
        "steps_before_run": start_step,
        "completed_steps": step,
        "steps_this_run": step - start_step,
        "batch_size": int(config["training"]["batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "losses": losses,
        "last_loss": losses[-1],
        "finite_loss": True,
        "tracked_parameter_max_abs_delta": delta,
        "checkpoint": str(final_checkpoint / "pretrained_model"),
        "checkpoint_model_sha256": sha256_file(final_checkpoint / "pretrained_model" / "model.safetensors"),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "frozen_vlm": frozen_vlm,
        "gradient_checkpointing": bool(config["model"].get("gradient_checkpointing", True)),
        "compile_model": bool(config["model"].get("compile_model", False)),
        "augmentation": asdict(augmentation),
        "bottleneck_size": int(dataset_cfg.get("bottleneck_size", 128)),
        "elapsed_s": elapsed,
        "steps_per_second": (step - start_step) / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None)
    args = parser.parse_args()
    train(args.config, args.output_dir, args.steps, args.hours)


if __name__ == "__main__":
    main()
