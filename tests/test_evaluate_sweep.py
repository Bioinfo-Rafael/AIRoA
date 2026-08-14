from pathlib import Path

import pytest

from airoa.pi05.evaluate_sweep import (
    aggregate_results,
    checkpoint_candidates,
    select_sample_refs,
    selection_proxy,
)


def _episode(task: int, episode: int, length: int = 120) -> dict:
    start = episode * 1000
    return {
        "episode_index": episode,
        "dataset_from_index": start,
        "dataset_to_index": start + length,
        "length": length,
        "tasks": [f"task {task}"],
        "stats/task_index/min": [task],
    }


def test_sample_selection_is_balanced_distinct_interior_and_deterministic():
    episodes = [_episode(task, task * 10 + episode) for task in range(3) for episode in range(4)]
    kwargs = {"samples_per_task": 2, "seed": 123, "expected_tasks": 3, "chunk_size": 50}
    first = select_sample_refs(episodes, {i: f"task {i}" for i in range(3)}, **kwargs)
    second = select_sample_refs(episodes, {i: f"task {i}" for i in range(3)}, **kwargs)
    assert first == second
    assert len(first) == 6
    for task in range(3):
        task_refs = [ref for ref in first if ref.task_index == task]
        assert len({ref.episode_index for ref in task_refs}) == 2
        assert all(0 < ref.frame_index < ref.episode_length - 50 for ref in task_refs)


def test_selection_proxy_uses_requested_formula():
    assert selection_proxy(0.25, 2.0, 8) == pytest.approx(0.2297)


def _sample_row(checkpoint: str, task: int, sample: int, mse: float, jerk: float = 1.0) -> dict:
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": f"/{checkpoint}",
        "training_step": None if checkpoint == "base" else int(checkpoint),
        "horizon": 5,
        "task_index": task,
        "task": f"task {task}",
        "sample_id": f"{task}-{sample}",
        "action_mse": mse,
        "xyz_action_jerk": jerk,
        "rotation_jerk": 0.5,
        "selection_proxy": selection_proxy(mse, jerk, 5),
        "inference_latency_s": 1.0 + sample,
    }


def test_aggregation_is_task_equal_weighted_and_compares_base_and_11500():
    rows = [
        _sample_row("base", 0, 0, 2.0),
        _sample_row("base", 0, 1, 2.0),
        _sample_row("base", 1, 0, 4.0),
        _sample_row("base", 1, 1, 4.0),
        _sample_row("011500", 0, 0, 1.0),
        _sample_row("011500", 0, 1, 1.0),
        _sample_row("011500", 1, 0, 3.0),
        _sample_row("011500", 1, 1, 3.0),
    ]
    benchmarks = [
        {"checkpoint": name, "status": "pass", "peak_cuda_memory_bytes": 10, "peak_cuda_memory_gib": 1.0}
        for name in ("base", "011500")
    ]
    per_task, summary = aggregate_results(rows, benchmarks)
    assert len(per_task) == 4
    by_checkpoint = {row["checkpoint"]: row for row in summary}
    assert by_checkpoint["base"]["mean_action_mse"] == pytest.approx(3.0)
    assert by_checkpoint["011500"]["mean_action_mse"] == pytest.approx(2.0)
    assert by_checkpoint["011500"]["delta_vs_base_mse"] == pytest.approx(-1.0)
    assert by_checkpoint["011500"]["relative_improvement_vs_base_percent"] == pytest.approx(100 / 3)
    assert by_checkpoint["base"]["delta_vs_ckpt11500_mse"] == pytest.approx(1.0)


def test_checkpoint_paths_match_requested_steps():
    rows = checkpoint_candidates(Path("outputs/pi05_track1"), [1000, 12131], Path("/base"))
    assert [row.name for row in rows] == ["base", "001000", "012131"]
    assert rows[-1].path == Path("outputs/pi05_track1/checkpoints/012131/pretrained_model")
