#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as error:
        return f"unavailable: {error}"
    return ((result.stdout or "") + (result.stderr or "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/environment"))
    parser.add_argument("--allow-no-gpu", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = command_output(["nvidia-smi"])
    details = {
        "python": sys.version,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": gpu_info,
    }
    (args.output_dir / "python_version.txt").write_text(sys.version + "\n")
    (args.output_dir / "torch_version.txt").write_text(torch.__version__ + "\n")
    (args.output_dir / "cuda_version.txt").write_text(str(torch.version.cuda) + "\n")
    (args.output_dir / "gpu_info.txt").write_text(gpu_info + "\n")
    (args.output_dir / "platform.json").write_text(json.dumps(details, indent=2) + "\n")
    freeze = command_output([sys.executable, "-m", "pip", "freeze", "--all"])
    (args.output_dir / "pip_freeze.txt").write_text(freeze + "\n")
    git_sha = command_output(["git", "rev-parse", "HEAD"])
    (args.output_dir / "git_commit.txt").write_text(git_sha + "\n")
    print(json.dumps(details, indent=2))
    if not torch.cuda.is_available() and not args.allow_no_gpu:
        raise SystemExit(
            "GPU unavailable: verify NVIDIA Container Toolkit, `docker run --gpus all`, and driver compatibility"
        )


if __name__ == "__main__":
    main()
