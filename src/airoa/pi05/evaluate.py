"""Real-batch checkpoint reload, inference, and small checkpoint/horizon gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from airoa.config import load_config
from airoa.constants import DATASET_REVISION, LEROBOT_REVISION, MODEL_ID, MODEL_REVISION
from airoa.data.libero_plus import PhotometricConfig, make_dataset
from airoa.metrics.smoothness import trajectory_metrics
from airoa.pi05.checkpoint import download_tokenizer, load_policy_and_processors, sha256_file


def _checkpoint_candidates(output_dir: Path, count: int) -> list[Path]:
    candidates = sorted(path / "pretrained_model" for path in (output_dir / "checkpoints").glob("[0-9]*"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints below {output_dir}")
    return candidates[-count:]


def _infer(checkpoint: Path, tokenizer: Path, sample: dict[str, Any], seed: int) -> tuple[np.ndarray, float, int]:
    policy, preprocessor, postprocessor = load_policy_and_processors(
        checkpoint,
        tokenizer,
        device="cuda",
        training=False,
        gradient_checkpointing=False,
        compile_model=False,
    )
    batch = preprocessor(sample)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    noise = torch.randn((1, 50, 32), generator=generator, device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(batch, noise=noise)
        actions = postprocessor(normalized)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    result = actions.detach().cpu().numpy().astype(np.float32)
    if result.shape != (1, 50, 7) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid inference output: shape={result.shape}, finite={np.isfinite(result).all()}")
    del policy, preprocessor, postprocessor, batch, normalized, actions
    torch.cuda.empty_cache()
    return result[0], latency, peak_memory


def evaluate(config_path: Path, output_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real checkpoint inference")
    config = load_config(config_path)
    seed = int(config.get("seed", 1000))
    dataset = make_dataset(
        episodes=[int(config["evaluation"].get("episode", 0))],
        video_backend=config["dataset"].get("video_backend", "torchcodec"),
        bottleneck_size=int(config["dataset"].get("bottleneck_size", 128)),
        restore_feature_resolution=bool(config["dataset"].get("restore_feature_resolution", True)),
        augmentation=PhotometricConfig(enabled=False),
    )
    sample = dataset[int(config["evaluation"].get("frame", 0))]
    target = sample["action"].detach().cpu().numpy().astype(np.float32)
    cache_dir = Path(os.environ.get("HF_HOME", "cache/huggingface"))
    tokenizer = download_tokenizer(cache_dir)
    rows: list[dict[str, Any]] = []
    benchmark: dict[str, Any] = {"candidates": []}
    for checkpoint in _checkpoint_candidates(output_dir, int(config["evaluation"].get("checkpoint_candidates", 3))):
        actions, latency, peak_memory = _infer(checkpoint, tokenizer, sample, seed)
        mse = float(np.mean((actions - target) ** 2))
        benchmark["candidates"].append(
            {
                "checkpoint": str(checkpoint),
                "action_shape": list(actions.shape),
                "dtype": str(actions.dtype),
                "finite": bool(np.isfinite(actions).all()),
                "latency_s": latency,
                "peak_cuda_memory_bytes": peak_memory,
            }
        )
        for horizon in config["evaluation"].get("horizons", [5, 8, 10]):
            metrics = trajectory_metrics(actions[: int(horizon)])
            proxy_score = mse + 0.01 * float(metrics["xyz_action_jerk"]) + 0.0001 * (int(horizon) - 5)
            rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "horizon": int(horizon),
                    "success": None,
                    "collision": None,
                    "action_mse": mse,
                    "inference_latency_s": latency,
                    "selection_proxy": proxy_score,
                    **metrics,
                }
            )
    selected = min(rows, key=lambda row: float(row["selection_proxy"]))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (artifacts_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Checkpoint / horizon proxy evaluation",
        "",
        "`success` and `collision` are intentionally null: this gate uses a held-out real LIBERO-Plus batch, not a simulator rollout.",
        "",
        "| checkpoint | horizon | action MSE | latency (s) | xyz jerk | rotation jerk | proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {Path(row['checkpoint']).parent.name} | {row['horizon']} | {row['action_mse']:.6f} | "
            f"{row['inference_latency_s']:.3f} | {row['xyz_action_jerk']:.6f} | "
            f"{row['rotation_jerk']:.6f} | {row['selection_proxy']:.6f} |"
        )
    (artifacts_dir / "evaluation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (artifacts_dir / "inference_benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    training_summary_path = output_dir / "training_summary.json"
    training_summary = json.loads(training_summary_path.read_text()) if training_summary_path.is_file() else {}
    selected_model = {
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": sha256_file(Path(selected["checkpoint"]) / "model.safetensors"),
        "pretrained_model": MODEL_ID,
        "pretrained_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "lerobot_revision": LEROBOT_REVISION,
        "execution_horizon": selected["horizon"],
        "model_settings": config["model"],
        "training_steps": training_summary.get("completed_steps"),
        "batch_size": config["training"]["batch_size"],
        "normalization": "MEAN_STD from checkpoint processor files",
        "augmentation": config["augmentation"],
        "selection": {
            "method": "held-out real LIBERO-Plus action MSE + smoothness gate",
            "success": None,
            "collision": None,
            "proxy_score": selected["selection_proxy"],
        },
    }
    (artifacts_dir / "selected_model.json").write_text(json.dumps(selected_model, indent=2) + "\n")
    return selected_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.output_dir, args.artifacts_dir), indent=2))


if __name__ == "__main__":
    main()
