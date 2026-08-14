"""Deterministic multi-task checkpoint and execution-horizon evaluation.

This module is inference-only. It never writes training checkpoints, selected_model.json,
or submission artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from airoa.constants import DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION
from airoa.data.libero_plus import PhotometricConfig, download_metadata, make_dataset
from airoa.metrics.smoothness import trajectory_metrics
from airoa.pi05.checkpoint import download_checkpoint, download_tokenizer, load_policy_and_processors

DEFAULT_CHECKPOINT_STEPS = (1000, 3000, 5000, 8000, 10000, 11500, 12131)
DEFAULT_HORIZONS = (5, 8, 10)
ACTION_CHUNK_SIZE = 50
ACTION_DIM = 7


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    task_index: int
    task: str
    sample_index_within_task: int
    episode_index: int
    episode_length: int
    frame_index: int
    absolute_index: int


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    training_step: int | None
    is_base: bool = False


def selection_proxy(action_mse: float, xyz_action_jerk: float, horizon: int) -> float:
    """Return the requested checkpoint/horizon selection proxy."""
    return float(action_mse - 0.01 * xyz_action_jerk - 0.0001 * (horizon - 5))


def _scalar(value: Any) -> int:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected a single-valued metadata statistic, got {value}")
        value = value[0]
    return int(value)


def select_sample_refs(
    episode_rows: Sequence[dict[str, Any]],
    task_names: dict[int, str],
    *,
    samples_per_task: int,
    seed: int,
    expected_tasks: int = 40,
    chunk_size: int = ACTION_CHUNK_SIZE,
) -> list[SampleRef]:
    """Select distinct episodes and interior frames uniformly across tasks."""
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        task_index = _scalar(row["stats/task_index/min"])
        length = int(row["length"])
        if length >= chunk_size + 1:
            by_task[task_index].append(row)
    if sorted(by_task) != list(range(expected_tasks)):
        raise ValueError(f"Expected task indices 0..{expected_tasks - 1}, found {sorted(by_task)}")

    refs: list[SampleRef] = []
    for task_index in range(expected_tasks):
        rows = sorted(by_task[task_index], key=lambda row: int(row["episode_index"]))
        if len(rows) < samples_per_task:
            raise ValueError(
                f"Task {task_index} has only {len(rows)} episodes with a full {chunk_size}-step target"
            )
        rng = random.Random(seed + 1_000_003 * task_index)
        chosen = rng.sample(rows, samples_per_task)
        for sample_index, row in enumerate(chosen):
            length = int(row["length"])
            latest_frame = length - chunk_size
            # Prefer the episode interior while preserving a complete, unpadded action target.
            margin = min(max(1, length // 10), latest_frame // 3)
            low = margin
            high = latest_frame - margin
            if high < low:
                low, high = 0, latest_frame
            frame_index = rng.randint(low, high)
            start = int(row["dataset_from_index"])
            refs.append(
                SampleRef(
                    sample_id=f"task_{task_index:02d}_sample_{sample_index:02d}",
                    task_index=task_index,
                    task=task_names.get(task_index, str(row.get("tasks", [task_index])[0])),
                    sample_index_within_task=sample_index,
                    episode_index=int(row["episode_index"]),
                    episode_length=length,
                    frame_index=frame_index,
                    absolute_index=start + frame_index,
                )
            )
    return refs


def build_sample_manifest(
    metadata_snapshot: Path, *, samples_per_task: int, seed: int
) -> list[SampleRef]:
    import pyarrow.dataset as arrow_dataset
    import pyarrow.parquet as parquet

    episode_table = arrow_dataset.dataset(
        str(metadata_snapshot / "meta" / "episodes"), format="parquet"
    ).to_table(
        columns=[
            "episode_index",
            "dataset_from_index",
            "dataset_to_index",
            "length",
            "tasks",
            "stats/task_index/min",
        ]
    )
    task_rows = parquet.read_table(metadata_snapshot / "meta" / "tasks.parquet").to_pylist()
    task_names = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    return select_sample_refs(
        episode_table.to_pylist(),
        task_names,
        samples_per_task=samples_per_task,
        seed=seed,
    )


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def load_samples(
    refs: Sequence[SampleRef],
    *,
    video_backend: str,
    bottleneck_size: int,
    restore_feature_resolution: bool,
) -> list[dict[str, Any]]:
    """Decode every selected observation once using the existing LeRobot/PyAV path."""
    episodes = sorted({ref.episode_index for ref in refs})
    dataset = make_dataset(
        episodes=episodes,
        video_backend=video_backend,
        bottleneck_size=bottleneck_size,
        restore_feature_resolution=restore_feature_resolution,
        augmentation=PhotometricConfig(enabled=False),
    )
    absolute_to_relative = getattr(dataset.dataset, "_absolute_to_relative_idx", None)
    if absolute_to_relative is None:
        absolute_to_relative = {
            _to_int(value): index for index, value in enumerate(dataset.dataset.hf_dataset["index"])
        }

    samples: list[dict[str, Any]] = []
    for ref in refs:
        if ref.absolute_index not in absolute_to_relative:
            raise KeyError(f"Selected dataset index was not loaded: {ref.absolute_index}")
        sample = dataset[int(absolute_to_relative[ref.absolute_index])]
        target = sample["action"]
        if tuple(target.shape) != (ACTION_CHUNK_SIZE, ACTION_DIM):
            raise ValueError(f"{ref.sample_id}: expected target [50,7], got {tuple(target.shape)}")
        padding = sample.get("action_is_pad")
        if padding is not None and bool(torch.as_tensor(padding).any()):
            raise ValueError(f"{ref.sample_id}: selected target contains padded actions")
        if _to_int(sample["episode_index"]) != ref.episode_index:
            raise ValueError(f"{ref.sample_id}: episode mapping changed during dataset load")
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError(f"{ref.sample_id}: target action contains non-finite values")
        samples.append(sample)
    return samples


def checkpoint_candidates(output_dir: Path, steps: Sequence[int], base_path: Path) -> list[Candidate]:
    candidates = [Candidate("base", base_path, None, is_base=True)]
    for step in steps:
        candidates.append(
            Candidate(
                name=f"{step:06d}",
                path=output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model",
                training_step=step,
            )
        )
    return candidates


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep the cached raw CPU observation immutable across checkpoint processors."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def _evaluate_candidate(
    candidate: Candidate,
    refs: Sequence[SampleRef],
    samples: Sequence[dict[str, Any]],
    tokenizer_dir: Path,
    *,
    horizons: Sequence[int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(refs) != len(samples):
        raise ValueError("Sample references and decoded samples differ in length")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    load_started = time.perf_counter()
    policy, preprocessor, postprocessor = load_policy_and_processors(
        candidate.path,
        tokenizer_dir,
        device="cuda",
        training=False,
        gradient_checkpointing=False,
        compile_model=False,
    )
    torch.cuda.synchronize()
    load_latency = time.perf_counter() - load_started
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    try:
        for ref, sample in zip(refs, samples, strict=True):
            batch = preprocessor(_clone_sample(sample))
            noise = torch.randn(
                (1, ACTION_CHUNK_SIZE, 32),
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(batch, noise=noise)
                predicted = postprocessor(normalized)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            actions = predicted.detach().cpu().numpy().astype(np.float32)
            if actions.shape != (1, ACTION_CHUNK_SIZE, ACTION_DIM) or not np.isfinite(actions).all():
                raise RuntimeError(
                    f"{candidate.name}/{ref.sample_id}: invalid output {actions.shape}, "
                    f"finite={np.isfinite(actions).all()}"
                )
            target = sample["action"].detach().cpu().numpy().astype(np.float32)
            action_mse = float(np.mean((actions[0] - target) ** 2))
            latencies.append(latency)
            for horizon in horizons:
                metrics = trajectory_metrics(actions[0, :horizon])
                rows.append(
                    {
                        "checkpoint": candidate.name,
                        "checkpoint_path": str(candidate.path),
                        "training_step": candidate.training_step,
                        "is_base": candidate.is_base,
                        **asdict(ref),
                        "horizon": horizon,
                        "action_mse": action_mse,
                        "selection_proxy": selection_proxy(
                            action_mse, float(metrics["xyz_action_jerk"]), horizon
                        ),
                        "inference_latency_s": latency,
                        **metrics,
                    }
                )
            del batch, noise, normalized, predicted, actions
        peak_memory = int(torch.cuda.max_memory_allocated())
        benchmark = {
            "checkpoint": candidate.name,
            "checkpoint_path": str(candidate.path),
            "training_step": candidate.training_step,
            "status": "pass",
            "model_load_count": 1,
            "model_load_latency_s": load_latency,
            "n_samples": len(samples),
            "inference_latency_mean_s": float(np.mean(latencies)),
            "inference_latency_median_s": float(np.median(latencies)),
            "inference_latency_first_s": latencies[0],
            "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_memory_gib": peak_memory / 1024**3,
            "action_shape": [ACTION_CHUNK_SIZE, ACTION_DIM],
            "finite": True,
        }
        return rows, benchmark
    finally:
        del policy, preprocessor, postprocessor
        gc.collect()
        torch.cuda.empty_cache()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values))


def _aggregate_per_task(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["checkpoint"]), int(row["horizon"]), int(row["task_index"]))].append(row)
    result: list[dict[str, Any]] = []
    for (checkpoint, horizon, task_index), group in groups.items():
        result.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_path": group[0]["checkpoint_path"],
                "training_step": group[0]["training_step"],
                "horizon": horizon,
                "task_index": task_index,
                "task": group[0]["task"],
                "n_samples": len(group),
                "mean_action_mse": _mean(float(row["action_mse"]) for row in group),
                "median_action_mse": float(median(float(row["action_mse"]) for row in group)),
                "mean_xyz_action_jerk": _mean(float(row["xyz_action_jerk"]) for row in group),
                "mean_rotation_jerk": _mean(float(row["rotation_jerk"]) for row in group),
                "mean_proxy": _mean(float(row["selection_proxy"]) for row in group),
                "median_proxy": float(median(float(row["selection_proxy"]) for row in group)),
                "inference_latency_mean_s": _mean(float(row["inference_latency_s"]) for row in group),
            }
        )
    return sorted(result, key=lambda row: (str(row["checkpoint"]), int(row["horizon"]), int(row["task_index"])))


def aggregate_results(
    rows: Sequence[dict[str, Any]], benchmarks: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_task = _aggregate_per_task(rows)
    grouped_samples: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    grouped_tasks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_samples[(str(row["checkpoint"]), int(row["horizon"]))].append(row)
    for row in per_task:
        grouped_tasks[(str(row["checkpoint"]), int(row["horizon"]))].append(row)
    benchmark_by_checkpoint = {str(row["checkpoint"]): row for row in benchmarks if row["status"] == "pass"}

    summary: list[dict[str, Any]] = []
    for key, sample_group in grouped_samples.items():
        checkpoint, horizon = key
        task_group = grouped_tasks[key]
        mse_values = np.asarray([float(row["action_mse"]) for row in sample_group], dtype=np.float64)
        proxy_values = np.asarray([float(row["selection_proxy"]) for row in sample_group], dtype=np.float64)
        benchmark = benchmark_by_checkpoint[checkpoint]
        summary.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_path": sample_group[0]["checkpoint_path"],
                "training_step": sample_group[0]["training_step"],
                "horizon": horizon,
                "n_samples": len(sample_group),
                "n_tasks": len(task_group),
                # Means are explicitly task-equal-weighted; medians/std/p90 use paired samples.
                "mean_action_mse": _mean(float(row["mean_action_mse"]) for row in task_group),
                "median_action_mse": float(np.median(mse_values)),
                "std_action_mse": float(np.std(mse_values)),
                "p90_action_mse": float(np.percentile(mse_values, 90)),
                "mean_xyz_action_jerk": _mean(
                    float(row["mean_xyz_action_jerk"]) for row in task_group
                ),
                "mean_rotation_jerk": _mean(float(row["mean_rotation_jerk"]) for row in task_group),
                "mean_proxy": _mean(float(row["mean_proxy"]) for row in task_group),
                "median_proxy": float(np.median(proxy_values)),
                "inference_latency_mean_s": _mean(
                    float(row["inference_latency_s"]) for row in sample_group
                ),
                "inference_latency_median_s": float(
                    np.median([float(row["inference_latency_s"]) for row in sample_group])
                ),
                "peak_cuda_memory_bytes": benchmark["peak_cuda_memory_bytes"],
                "peak_cuda_memory_gib": benchmark["peak_cuda_memory_gib"],
            }
        )

    by_key = {(str(row["checkpoint"]), int(row["horizon"])): row for row in summary}
    for row in summary:
        horizon = int(row["horizon"])
        _add_comparison(row, by_key.get(("base", horizon)), "base")
        _add_comparison(row, by_key.get(("011500", horizon)), "ckpt11500")
    summary.sort(key=lambda row: (float(row["mean_proxy"]), float(row["mean_action_mse"])))
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return per_task, summary


def _add_comparison(
    row: dict[str, Any], reference: dict[str, Any] | None, suffix: str
) -> None:
    if reference is None:
        row[f"delta_vs_{suffix}_mse"] = None
        row[f"relative_improvement_vs_{suffix}_percent"] = None
        return
    value = float(row["mean_action_mse"])
    baseline = float(reference["mean_action_mse"])
    row[f"delta_vs_{suffix}_mse"] = value - baseline
    row[f"relative_improvement_vs_{suffix}_percent"] = (
        100.0 * (baseline - value) / baseline if baseline else None
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value: Any, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _summary_markdown(
    summary: Sequence[dict[str, Any]],
    benchmarks: Sequence[dict[str, Any]],
    *,
    samples_per_task: int,
    seed: int,
) -> str:
    lines = [
        "# π0.5 checkpoint sweep",
        "",
        f"Dataset: `{DATASET_ID}` @ `{DATASET_REVISION}`  ",
        f"Selection: 40 task-equal-weighted samples, {samples_per_task}/task, seed {seed}, augmentation off.  ",
        "Proxy (lower ranks first): `action_mse - 0.01 * xyz_action_jerk - 0.0001 * (horizon - 5)`.",
        "",
        "## Ranking",
        "",
        "| rank | checkpoint | horizon | mean MSE | median MSE | mean xyz jerk | mean proxy | vs base |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['rank']} | {row['checkpoint']} | {row['horizon']} | "
            f"{row['mean_action_mse']:.6f} | {row['median_action_mse']:.6f} | "
            f"{row['mean_xyz_action_jerk']:.6f} | {row['mean_proxy']:.6f} | "
            f"{_format_optional(row['relative_improvement_vs_base_percent'], 2)}% |"
        )

    lines.extend(
        [
            "",
            "## MSE progression by training step",
            "",
            "MSE does not depend on execution horizon, so horizon 5 is shown once per checkpoint.",
            "",
            "| checkpoint | step | mean MSE | delta vs base | relative improvement vs base |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    progression = [row for row in summary if int(row["horizon"]) == 5]
    progression.sort(key=lambda row: -1 if row["training_step"] is None else int(row["training_step"]))
    for row in progression:
        step = "base" if row["training_step"] is None else str(row["training_step"])
        lines.append(
            f"| {row['checkpoint']} | {step} | {row['mean_action_mse']:.6f} | "
            f"{_format_optional(row['delta_vs_base_mse'])} | "
            f"{_format_optional(row['relative_improvement_vs_base_percent'], 2)}% |"
        )

    missing = [row for row in benchmarks if row["status"] == "missing"]
    failed = [row for row in benchmarks if row["status"] == "failed"]
    lines.extend(["", "## Candidate status", ""])
    if not missing and not failed:
        lines.append("All requested candidates completed.")
    for row in missing:
        lines.append(f"- Missing `{row['checkpoint']}`: `{row['checkpoint_path']}`")
    for row in failed:
        lines.append(f"- Failed `{row['checkpoint']}`: `{row['error_type']}: {row['error']}`")
    return "\n".join(lines) + "\n"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_sweep(
    *,
    output_dir: Path,
    artifacts_dir: Path,
    samples_per_task: int,
    checkpoint_steps: Sequence[int],
    horizons: Sequence[int],
    seed: int,
    video_backend: str,
    bottleneck_size: int,
    restore_feature_resolution: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real π0.5 checkpoint sweep evaluation")
    if tuple(sorted(set(horizons))) != tuple(horizons) or any(h < 3 or h > ACTION_CHUNK_SIZE for h in horizons):
        raise ValueError("Horizons must be unique, sorted, and between 3 and 50")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    metadata_snapshot = download_metadata()
    refs = build_sample_manifest(metadata_snapshot, samples_per_task=samples_per_task, seed=seed)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(
        artifacts_dir / "samples.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "selection_seed": seed,
            "samples_per_task": samples_per_task,
            "n_tasks": len({ref.task_index for ref in refs}),
            "n_samples": len(refs),
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "video_backend": video_backend,
            "augmentation": False,
            "samples": [asdict(ref) for ref in refs],
        },
    )
    samples = load_samples(
        refs,
        video_backend=video_backend,
        bottleneck_size=bottleneck_size,
        restore_feature_resolution=restore_feature_resolution,
    )

    cache_dir = Path(os.environ.get("HF_HOME", "cache/huggingface"))
    tokenizer_dir = download_tokenizer(cache_dir)
    candidates = checkpoint_candidates(output_dir, checkpoint_steps, download_checkpoint())
    all_rows: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.path.is_dir():
            report = {
                "checkpoint": candidate.name,
                "checkpoint_path": str(candidate.path),
                "training_step": candidate.training_step,
                "status": "missing",
            }
            benchmarks.append(report)
            print(json.dumps(report), flush=True)
            continue
        print(json.dumps({"checkpoint": candidate.name, "status": "loading"}), flush=True)
        try:
            candidate_rows, benchmark = _evaluate_candidate(
                candidate,
                refs,
                samples,
                tokenizer_dir,
                horizons=horizons,
                seed=seed,
            )
        except Exception as error:
            gc.collect()
            torch.cuda.empty_cache()
            report = {
                "checkpoint": candidate.name,
                "checkpoint_path": str(candidate.path),
                "training_step": candidate.training_step,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            benchmarks.append(report)
            print(json.dumps(report), flush=True)
            continue
        all_rows.extend(candidate_rows)
        benchmarks.append(benchmark)
        print(json.dumps(benchmark), flush=True)

    if not all_rows:
        _json_dump(artifacts_dir / "inference_benchmark.json", {"candidates": benchmarks})
        raise RuntimeError("No checkpoint completed the sweep")
    per_task, summary = aggregate_results(all_rows, benchmarks)

    _write_csv(artifacts_dir / "per_sample.csv", all_rows)
    _write_csv(artifacts_dir / "per_task.csv", per_task)
    _write_csv(artifacts_dir / "summary.csv", summary)
    (artifacts_dir / "summary.md").write_text(
        _summary_markdown(summary, benchmarks, samples_per_task=samples_per_task, seed=seed),
        encoding="utf-8",
    )
    _json_dump(
        artifacts_dir / "inference_benchmark.json",
        {
            "device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "noise_seed": seed,
            "noise_policy": "one identically seeded CUDA generator sequence per checkpoint",
            "compile_model": False,
            "candidates": benchmarks,
        },
    )
    if not any(row["checkpoint"] == "base" for row in summary):
        raise RuntimeError("Base pretrained evaluation failed; improvement comparisons are unavailable")
    best = summary[0]
    best_document = {
        "selected_checkpoint": best["checkpoint_path"],
        "selected_checkpoint_name": best["checkpoint"],
        "selected_horizon": best["horizon"],
        "mean_mse": best["mean_action_mse"],
        "mean_proxy": best["mean_proxy"],
        "relative_improvement_vs_base_percent": best["relative_improvement_vs_base_percent"],
        "samples_per_task": samples_per_task,
        "n_tasks": best["n_tasks"],
        "n_samples": best["n_samples"],
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "seed": seed,
    }
    _json_dump(artifacts_dir / "best_checkpoint.json", best_document)
    return best_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi05_track1"))
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/checkpoint_sweep"))
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--bottleneck-size", type=int, default=128)
    parser.add_argument(
        "--no-restore-feature-resolution", action="store_false", dest="restore_feature_resolution"
    )
    parser.set_defaults(restore_feature_resolution=True)
    args = parser.parse_args()
    result = evaluate_sweep(
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        samples_per_task=args.samples_per_task,
        checkpoint_steps=args.checkpoint_steps,
        horizons=args.horizons,
        seed=args.seed,
        video_backend=args.video_backend,
        bottleneck_size=args.bottleneck_size,
        restore_feature_resolution=args.restore_feature_resolution,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
