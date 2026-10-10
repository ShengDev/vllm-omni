# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in per-phase wall timers for Noisy PP tick loop (bottleneck evidence).

Enable with::

    AR_DIFFUSION_PHASE_PROFILE_DIR=/path/to/dir

Each rank writes ``phase_rank{R}.json`` at pipeline end. Host timers capture
CPU-side stalls (Gloo metadata in ``isend_tensor_dict``, NCCL ``wait``).
Every ``AR_DIFFUSION_PHASE_GPU_SAMPLE`` active ticks (default 30), also
cuda-synchronizes around ``forward`` to sample true GPU block time.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_CURRENT: PhaseProfiler | None = None
_DISABLED: PhaseProfiler | None = None


def profile_dir() -> Path | None:
    raw = os.environ.get("AR_DIFFUSION_PHASE_PROFILE_DIR")
    if not raw:
        return None
    return Path(raw)


def gpu_sample_period() -> int:
    try:
        return max(1, int(os.environ.get("AR_DIFFUSION_PHASE_GPU_SAMPLE", "30")))
    except ValueError:
        return 30


@dataclass
class PhaseProfiler:
    rank: int
    enabled: bool = False
    host_s: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _active_ticks: int = 0
    _gpu_samples: list[float] = field(default_factory=list)

    @classmethod
    def maybe(cls, rank: int) -> PhaseProfiler:
        d = profile_dir()
        if d is None:
            return cls(rank=rank, enabled=False)
        d.mkdir(parents=True, exist_ok=True)
        return cls(rank=rank, enabled=True)

    def add(self, name: str, seconds: float, *, n: int = 1) -> None:
        if not self.enabled:
            return
        self.host_s[name] += float(seconds)
        self.counts[name] += int(n)

    def add_total(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self.totals[name] += float(value)

    def note_active_tick(self) -> bool:
        """Return True when this active tick should sample GPU forward time."""
        if not self.enabled:
            return False
        self._active_ticks += 1
        self.counts["active_ticks"] += 1
        return (self._active_ticks % gpu_sample_period()) == 0

    def record_gpu_sample(self, seconds: float) -> None:
        if not self.enabled:
            return
        self._gpu_samples.append(float(seconds))
        self.host_s["forward_gpu_sampled"] += float(seconds)
        self.counts["forward_gpu_samples"] += 1

    def dump(self, extra: dict[str, Any] | None = None) -> Path | None:
        if not self.enabled:
            return None
        d = profile_dir()
        assert d is not None
        host = dict(self.host_s)
        total_host = sum(host.values()) or 1.0
        frac = {k: v / total_host for k, v in sorted(host.items(), key=lambda kv: -kv[1])}
        samples = list(self._gpu_samples)
        report = {
            "rank": self.rank,
            "host_seconds": host,
            "host_fraction_of_summed_phases": frac,
            "counts": dict(self.counts),
            "totals": dict(self.totals),
            "forward_gpu_sample_seconds": {
                "n": len(samples),
                "mean": (sum(samples) / len(samples)) if samples else None,
                "min": min(samples) if samples else None,
                "max": max(samples) if samples else None,
                "sum": sum(samples) if samples else 0.0,
            },
            "note": (
                "host_seconds are wall times around each phase on this rank. "
                "kv_exchange includes blocking Gloo metadata in isend/irecv_tensor_dict. "
                "kv_pack/kv_gloo_meta/kv_nccl_post are sub-phases inside exchange when enabled. "
                "forward_host is usually launch time; forward_gpu_sampled uses cuda "
                "synchronize around forward every AR_DIFFUSION_PHASE_GPU_SAMPLE active ticks."
            ),
            "extra": extra or {},
        }
        path = d / f"phase_rank{self.rank}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        return path


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


@contextmanager
def timed(profiler: PhaseProfiler, name: str) -> Iterator[None]:
    if not profiler.enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        profiler.add(name, time.perf_counter() - t0)


def measure_forward(
    profiler: PhaseProfiler,
    forward_fn: Callable[[], T],
    *,
    sample_gpu: bool,
) -> T:
    """Run forward; optionally sync to sample GPU time."""
    if not profiler.enabled:
        return forward_fn()
    if sample_gpu:
        _cuda_sync()
        t0 = time.perf_counter()
        out = forward_fn()
        _cuda_sync()
        dt = time.perf_counter() - t0
        profiler.add("forward_host", dt)
        profiler.record_gpu_sample(dt)
        return out
    t0 = time.perf_counter()
    out = forward_fn()
    profiler.add("forward_host", time.perf_counter() - t0)
    return out


@contextmanager
def bind_profiler(profiler: PhaseProfiler) -> Iterator[PhaseProfiler]:
    global _CURRENT
    prev = _CURRENT
    _CURRENT = profiler
    try:
        yield profiler
    finally:
        _CURRENT = prev


def current_profiler() -> PhaseProfiler:
    global _DISABLED
    if _CURRENT is not None:
        return _CURRENT
    if _DISABLED is None:
        _DISABLED = PhaseProfiler(rank=-1, enabled=False)
    return _DISABLED
