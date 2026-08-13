"""Offline π0.5 runtime with explicit action chunking for the PARC template."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from airoa.parc.adapter import parc_observation_to_pi05


class Pi05ParcPolicy:
    def __init__(self, model_dir: str | Path, execution_horizon: int | None = None) -> None:
        self.model_dir = Path(model_dir).resolve()
        runtime_config_path = self.model_dir / "runtime_config.json"
        runtime_config = (
            json.loads(runtime_config_path.read_text(encoding="utf-8"))
            if runtime_config_path.is_file()
            else {}
        )
        configured = execution_horizon or int(
            os.environ.get("AIROA_EXECUTION_HORIZON", runtime_config.get("execution_horizon", 5))
        )
        if configured not in (5, 8, 10):
            raise ValueError(f"execution_horizon must be 5, 8, or 10; got {configured}")
        self.execution_horizon = configured
        self.backend = os.environ.get("AIROA_POLICY_BACKEND", "pi05")
        self.action_queue: deque[np.ndarray] = deque()
        self.previous_chunk: np.ndarray | None = None
        self.instruction = ""
        self.last_inference_latency_s: float | None = None
        self._seed = 0
        if self.backend == "stub":
            self.policy = self.preprocessor = self.postprocessor = None
            self.device = "cpu"
        elif self.backend == "pi05":
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            if not (self.model_dir / "model.safetensors").is_file():
                raise FileNotFoundError(f"Offline model weights are missing: {self.model_dir}")
            tokenizer = self.model_dir / "tokenizer"
            if not tokenizer.is_dir():
                raise FileNotFoundError(f"Offline tokenizer is missing: {tokenizer}")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("π0.5 submission requires the PARC CUDA GPU")
            from airoa.pi05.checkpoint import load_policy_and_processors

            self.device = "cuda"
            self.policy, self.preprocessor, self.postprocessor = load_policy_and_processors(
                self.model_dir,
                tokenizer,
                device=self.device,
                training=False,
                gradient_checkpointing=False,
                compile_model=False,
            )
            self.generator = torch.Generator(device=self.device)
        else:
            raise ValueError(f"Unknown AIROA_POLICY_BACKEND={self.backend}")
        self.reset("")

    def _stub_chunk(self) -> np.ndarray:
        rows = [np.full(7, index / 100.0, dtype=np.float32) for index in range(50)]
        return np.stack(rows)

    def _infer_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self.backend == "stub":
            return self._stub_chunk()
        import torch

        prepared = parc_observation_to_pi05(obs, self.instruction)
        batch = self.preprocessor(prepared)
        noise = torch.randn(
            (1, 50, 32), generator=self.generator, device=self.device, dtype=torch.float32
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch, noise=noise)
            actions = self.postprocessor(normalized)
        torch.cuda.synchronize()
        self.last_inference_latency_s = time.perf_counter() - started
        chunk = actions.detach().cpu().numpy()[0].astype(np.float32)
        if chunk.shape != (50, 7) or not np.isfinite(chunk).all():
            raise RuntimeError(f"π0.5 returned invalid action chunk {chunk.shape}")
        return np.clip(chunk, -1.0, 1.0).astype(np.float32)

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if not self.action_queue:
            chunk = self._infer_chunk(obs)
            self.previous_chunk = chunk.copy()
            self.action_queue.extend(chunk[: self.execution_horizon])
        action = np.asarray(self.action_queue.popleft(), dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid queued action: {action}")
        return action

    def reset(self, instruction: str = "") -> None:
        self.action_queue.clear()
        self.previous_chunk = None
        self.instruction = str(instruction)
        self.last_inference_latency_s = None
        self._seed = int.from_bytes(hashlib.sha256(self.instruction.encode()).digest()[:8], "little")
        if self.backend == "pi05":
            self.generator.manual_seed(self._seed)
            self.policy.reset()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "execution_horizon": self.execution_horizon,
            "queued_actions": len(self.action_queue),
            "instruction": self.instruction,
            "last_inference_latency_s": self.last_inference_latency_s,
        }
