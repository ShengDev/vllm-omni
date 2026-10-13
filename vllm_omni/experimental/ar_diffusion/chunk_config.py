# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deploy / request helpers for Chunk Noisy PP (model-agnostic)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    ChunkPlan,
    ChunkSchedule,
    Ordering,
    build_chunk_plan,
)


def resolve_stage_config(od_config: Any) -> dict:
    """Merge ``ar_diffusion_stage_config`` from od_config and nested model_config."""
    stage = getattr(od_config, "ar_diffusion_stage_config", None)
    out: dict = dict(stage) if isinstance(stage, dict) else {}
    model_config = getattr(od_config, "model_config", None) or {}
    if isinstance(model_config, dict):
        nested = model_config.get("ar_diffusion_stage_config")
        if isinstance(nested, dict):
            out = {**out, **nested}
    return out


def resolve_enable_noisy_pp(od_config: Any) -> bool:
    """Opt-in flag for Chunk Noisy PP (default off).

    Checks ``ar_diffusion_stage_config.enable_noisy_pp``, then top-level
    ``model_config.enable_noisy_pp``, then nested stage config under model_config.
    """
    stage = getattr(od_config, "ar_diffusion_stage_config", None)
    if isinstance(stage, dict) and "enable_noisy_pp" in stage:
        return bool(stage["enable_noisy_pp"])
    model_config = getattr(od_config, "model_config", None) or {}
    if isinstance(model_config, dict):
        if "enable_noisy_pp" in model_config:
            return bool(model_config["enable_noisy_pp"])
        nested = model_config.get("ar_diffusion_stage_config")
        if isinstance(nested, dict) and "enable_noisy_pp" in nested:
            return bool(nested["enable_noisy_pp"])
    return False


def parse_chunk_ordering(name: Any) -> Ordering:
    schedule_name = str(name if name is not None else "serial").strip().lower()
    if schedule_name in ("", "serial"):
        return Ordering.SERIAL
    if schedule_name == "latest":
        return Ordering.INTERLEAVED
    raise ValueError(f"chunk_schedule must be 'serial' or 'latest', got {schedule_name!r}")


def build_plan_from_extra_args(
    extra_args: Mapping[str, Any] | None,
    *,
    denoise_steps: int,
    stages: int,
    layer_groups: int,
    blocks: int,
    max_history_chunks: int = 6,
) -> ChunkPlan:
    """Build a ``ChunkPlan`` from request ``extra_args`` + topology.

    Expected keys (optional): ``num_chunks`` / ``chunks``, ``kv_history_chunks``,
    ``chunk_schedule`` (``serial`` | ``latest``).
    """
    if denoise_steps < 1:
        raise ValueError(f"denoise_steps must be positive, got {denoise_steps}")
    if blocks < 1:
        raise ValueError(f"blocks must be positive, got {blocks}")
    if layer_groups < 1:
        raise ValueError(f"layer_groups must be positive, got {layer_groups}")
    if stages not in (1, denoise_steps + 1):
        raise ValueError(
            f"stage_parallel_size must be 1 or num_denoise_steps+1={denoise_steps + 1}, got {stages}"
        )

    args = dict(extra_args or {})
    chunks = int(args.get("num_chunks", args.get("chunks", 1)))
    history = int(args.get("kv_history_chunks", 0) or 0)
    if stages > 1 and history < 1:
        history = max(1, int(max_history_chunks))
    ordering = parse_chunk_ordering(args.get("chunk_schedule", "serial"))
    schedule = ChunkSchedule(
        chunks=chunks,
        num_denoise_steps=denoise_steps,
        stages=stages,
        layer_groups=layer_groups,
        ordering=ordering,
        kv_history_chunks=history,
        blocks=blocks,
        blocks_per_rank=blocks if layer_groups == 1 else None,
    )
    return build_chunk_plan(schedule)
