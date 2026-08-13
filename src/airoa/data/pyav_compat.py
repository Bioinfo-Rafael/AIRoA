"""PyAV decoder compatibility layer for pinned LeRobot v0.4.4 + torchvision >=0.26.

Torchvision 0.26 removed torchvision.io.VideoReader.
This backports the direct-PyAV approach used by newer LeRobot versions.
"""

from __future__ import annotations

from pathlib import Path

import av
import torch


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    if backend not in (None, "pyav", "video_reader"):
        raise ValueError(
            f"PyAV compatibility decoder received unsupported backend={backend!r}"
        )

    video_path = str(video_path)

    if not timestamps:
        raise ValueError("timestamps must not be empty")

    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]

        # Seek to nearest preceding keyframe and decode forward.
        offset = max(0, round(first_ts / stream.time_base) - 1)
        container.seek(
            offset,
            backward=True,
            any_frame=False,
            stream=stream,
        )

        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            current_ts = float(frame.pts * stream.time_base)

            arr = frame.to_ndarray(format="rgb24")
            tensor = (
                torch.from_numpy(arr)
                .permute(2, 0, 1)
                .contiguous()
            )

            loaded_frames.append(tensor)
            loaded_ts.append(current_ts)

            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise RuntimeError(
            f"No frames decoded from {video_path} "
            f"for timestamp range [{first_ts}, {last_ts}]"
        )

    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float64)

    dist = torch.cdist(
        query_ts[:, None],
        loaded_ts_tensor[:, None],
        p=1,
    )

    min_dist, argmin = dist.min(dim=1)

    within = min_dist <= tolerance_s
    if not bool(within.all()):
        raise RuntimeError(
            "Decoded frame timestamp exceeded tolerance. "
            f"min_dist={min_dist.tolist()} "
            f"tolerance_s={tolerance_s} "
            f"query={timestamps} "
            f"loaded={loaded_ts} "
            f"video={video_path}"
        )

    frames = torch.stack(
        [loaded_frames[int(i)] for i in argmin],
        dim=0,
    )

    if len(frames) != len(timestamps):
        raise RuntimeError(
            f"Expected {len(timestamps)} frames, got {len(frames)}"
        )

    return frames.to(torch.float32) / 255.0


def install_pyav_decoder_compat() -> None:
    """Patch the function reference imported by pinned LeRobotDataset."""

    import lerobot.datasets.lerobot_dataset as dataset_module
    import lerobot.datasets.video_utils as video_utils

    def dispatch(
        video_path,
        timestamps,
        tolerance_s,
        backend=None,
    ):
        if backend in ("pyav", "video_reader"):
            return decode_video_frames_pyav(
                video_path,
                timestamps,
                tolerance_s,
                backend,
            )

        # This compatibility path is intentionally only for PyAV.
        # AIRoA native remote configs use video_backend=pyav.
        raise ValueError(
            f"AIRoA PyAV compatibility decoder received backend={backend!r}"
        )

    # lerobot_dataset.py imports decode_video_frames directly,
    # therefore patch both the source module and imported reference.
    video_utils.decode_video_frames = dispatch
    dataset_module.decode_video_frames = dispatch
