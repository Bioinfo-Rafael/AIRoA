"""Pinned LIBERO-Plus access and Track 1-oriented image transforms."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_f
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as vision_f

from airoa.constants import (
    ACTION_KEY,
    DATASET_ID,
    DATASET_REVISION,
    FRONT_DATASET_KEY,
    STATE_KEY,
    WRIST_DATASET_KEY,
)


@dataclass(frozen=True)
class PhotometricConfig:
    enabled: bool = True
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.10
    hue: float = 0.025
    gamma: float = 0.10
    sharpness: float = 0.0


class Track1ImageTransform:
    """Correlated two-camera photometrics plus a 128-pixel information bottleneck."""

    def __init__(
        self,
        bottleneck_size: int = 128,
        restore_feature_resolution: bool = True,
        photometric: PhotometricConfig | None = None,
    ) -> None:
        self.bottleneck_size = bottleneck_size
        self.restore_feature_resolution = restore_feature_resolution
        self.photometric = photometric or PhotometricConfig()

    @staticmethod
    def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return torch_f.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear", align_corners=False, antialias=True
        ).squeeze(0)

    def _sample_params(self) -> dict[str, float]:
        p = self.photometric
        return {
            "brightness": random.uniform(1 - p.brightness, 1 + p.brightness),
            "contrast": random.uniform(1 - p.contrast, 1 + p.contrast),
            "saturation": random.uniform(1 - p.saturation, 1 + p.saturation),
            "hue": random.uniform(-p.hue, p.hue),
            "gamma": random.uniform(1 - p.gamma, 1 + p.gamma),
            "sharpness": random.uniform(1 - p.sharpness, 1 + p.sharpness) if p.sharpness else 1.0,
        }

    @staticmethod
    def _photometric(image: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        image = vision_f.adjust_brightness(image, params["brightness"])
        image = vision_f.adjust_contrast(image, params["contrast"])
        image = vision_f.adjust_saturation(image, params["saturation"])
        image = vision_f.adjust_hue(image, params["hue"])
        image = vision_f.adjust_gamma(image.clamp(0, 1), params["gamma"])
        if params["sharpness"] != 1.0:
            image = vision_f.adjust_sharpness(image, params["sharpness"])
        return image.clamp(0, 1)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        cameras = [FRONT_DATASET_KEY, WRIST_DATASET_KEY]
        original_sizes: dict[str, tuple[int, int]] = {}
        for key in cameras:
            image = result[key]
            if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"{key} must be a CHW torch tensor, got {type(image)} {getattr(image, 'shape', None)}")
            original_sizes[key] = (int(image.shape[-2]), int(image.shape[-1]))
            result[key] = self._resize(image, (self.bottleneck_size, self.bottleneck_size))

        if self.photometric.enabled:
            params = self._sample_params()
            for key in cameras:
                result[key] = self._photometric(result[key], params)

        if self.restore_feature_resolution:
            for key in cameras:
                result[key] = self._resize(result[key], original_sizes[key])
        return result


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Track1ImageTransform | None = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        return self.transform(sample) if self.transform else sample

    def __getattr__(self, name: str) -> Any:
        if name in {"dataset", "transform"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


class TaskBalancedSampler(Sampler[int]):
    """Samples all 40 primitives uniformly without inventing perturbation labels."""

    def __init__(self, dataset_root: str | Path, num_samples: int, seed: int = 1000) -> None:
        import pyarrow.dataset as arrow_dataset

        episode_dir = Path(dataset_root) / "meta" / "episodes"
        table = arrow_dataset.dataset(str(episode_dir), format="parquet").to_table(
            columns=["dataset_from_index", "dataset_to_index", "stats/task_index/min"]
        )
        values = table.to_pydict()
        self.by_task: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for start, stop, task_value in zip(
            values["dataset_from_index"],
            values["dataset_to_index"],
            values["stats/task_index/min"],
            strict=True,
        ):
            task = int(task_value[0] if isinstance(task_value, list) else task_value)
            self.by_task[task].append((int(start), int(stop)))
        if len(self.by_task) != 40:
            raise ValueError(f"Expected 40 LIBERO-Plus tasks, found {len(self.by_task)}")
        max_index = max(stop for intervals in self.by_task.values() for _, stop in intervals)
        if max_index != num_samples:
            raise ValueError(
                "TaskBalancedSampler currently requires the complete dataset; "
                f"metadata ends at {max_index}, dataset has {num_samples} frames"
            )
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        tasks = sorted(self.by_task)
        for _ in range(self.num_samples):
            episode = rng.choice(self.by_task[rng.choice(tasks)])
            yield rng.randrange(*episode)
        self.epoch += 1

    def __len__(self) -> int:
        return self.num_samples


def download_metadata(cache_dir: str | Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=["meta/*.json", "meta/*.parquet", "meta/episodes/**/*.parquet"],
        )
    )


def inspect_metadata(snapshot: str | Path) -> dict[str, Any]:
    import pyarrow.dataset as arrow_dataset
    import pyarrow.parquet as pq

    snapshot = Path(snapshot)
    info = json.loads((snapshot / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = pq.read_table(snapshot / "meta" / "tasks.parquet").to_pydict()
    episode_table = arrow_dataset.dataset(str(snapshot / "meta" / "episodes"), format="parquet").to_table(
        columns=["episode_index", "length", "stats/task_index/min"]
    )
    episodes = episode_table.to_pydict()
    task_ids = [int(value[0] if isinstance(value, list) else value) for value in episodes["stats/task_index/min"]]
    episode_counts = Counter(task_ids)
    feature_info = info["features"]
    candidate_perturbation_columns = [
        field.name
        for field in arrow_dataset.dataset(str(snapshot / "meta" / "episodes"), format="parquet").schema
        if any(token in field.name.lower() for token in ("perturb", "texture", "light", "background", "table"))
    ]
    return {
        "repo_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "codebase_version": info["codebase_version"],
        "camera_keys": [key for key, feature in feature_info.items() if feature["dtype"] == "video"],
        "state_shape": feature_info[STATE_KEY]["shape"],
        "action_shape": feature_info[ACTION_KEY]["shape"],
        "fps": info["fps"],
        "total_tasks": info["total_tasks"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "task_table_rows": len(tasks["task_index"]),
        "episode_count_by_task": {str(key): episode_counts[key] for key in sorted(episode_counts)},
        "perturbation_metadata_columns": candidate_perturbation_columns,
        "perturbation_balancing": (
            "metadata" if candidate_perturbation_columns else "disabled_no_reliable_metadata; full dataset + task balancing"
        ),
    }


def make_dataset(
    *,
    episodes: Sequence[int] | None,
    video_backend: str,
    bottleneck_size: int,
    restore_feature_resolution: bool,
    augmentation: PhotometricConfig,
    tolerance_s: float = 1e-4,
) -> TransformedDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(DATASET_ID, revision=DATASET_REVISION)
    delta_timestamps = {ACTION_KEY: [index / metadata.fps for index in range(50)]}
    dataset = LeRobotDataset(
        DATASET_ID,
        episodes=list(episodes) if episodes is not None else None,
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
        video_backend=video_backend,
        tolerance_s=tolerance_s,
    )
    transform = Track1ImageTransform(
        bottleneck_size=bottleneck_size,
        restore_feature_resolution=restore_feature_resolution,
        photometric=augmentation,
    )
    return TransformedDataset(dataset, transform)


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (FRONT_DATASET_KEY, WRIST_DATASET_KEY, STATE_KEY, ACTION_KEY):
        value = sample[key]
        summary[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all()),
            "min": float(value.min()),
            "max": float(value.max()),
        }
    summary["task"] = sample.get("task")
    summary["episode_index"] = int(sample["episode_index"])
    return summary
