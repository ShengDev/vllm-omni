# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-agnostic Chunk Noisy PP deploy / request helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.experimental.ar_diffusion.chunk_config import (
    build_plan_from_extra_args,
    parse_chunk_ordering,
    resolve_enable_noisy_kv,
    resolve_enable_noisy_pp,
    resolve_stage_config,
)
from vllm_omni.experimental.ar_diffusion.chunk_schedule import Ordering

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_enable_noisy_kv_defaults_off():
    od = SimpleNamespace(ar_diffusion_stage_config=None, model_config={})
    assert resolve_enable_noisy_kv(od) is False
    assert resolve_enable_noisy_pp(od) is False


def test_resolve_enable_noisy_kv_preferred_key():
    assert resolve_enable_noisy_kv(
        SimpleNamespace(ar_diffusion_stage_config={"enable_noisy_kv": True}, model_config={})
    )
    assert resolve_enable_noisy_kv(
        SimpleNamespace(ar_diffusion_stage_config=None, model_config={"enable_noisy_kv": True})
    )
    assert resolve_enable_noisy_kv(
        SimpleNamespace(
            ar_diffusion_stage_config=None,
            model_config={"ar_diffusion_stage_config": {"enable_noisy_kv": True}},
        )
    )


def test_resolve_enable_noisy_kv_legacy_alias():
    assert resolve_enable_noisy_kv(
        SimpleNamespace(ar_diffusion_stage_config={"enable_noisy_pp": True}, model_config={})
    )
    assert resolve_enable_noisy_pp(
        SimpleNamespace(ar_diffusion_stage_config=None, model_config={"enable_noisy_pp": True})
    )


def test_enable_noisy_kv_wins_over_legacy_alias():
    od = SimpleNamespace(
        ar_diffusion_stage_config={"enable_noisy_kv": False, "enable_noisy_pp": True},
        model_config={},
    )
    assert resolve_enable_noisy_kv(od) is False


def test_resolve_stage_config_merges_nested():
    od = SimpleNamespace(
        ar_diffusion_stage_config={"stage_parallel_size": 1, "max_history_chunks": 2},
        model_config={"ar_diffusion_stage_config": {"stage_parallel_size": 5}},
    )
    cfg = resolve_stage_config(od)
    assert cfg["stage_parallel_size"] == 5
    assert cfg["max_history_chunks"] == 2


def test_parse_chunk_ordering():
    assert parse_chunk_ordering("serial") is Ordering.SERIAL
    assert parse_chunk_ordering("latest") is Ordering.INTERLEAVED
    with pytest.raises(ValueError, match="chunk_schedule"):
        parse_chunk_ordering("vs")


def test_build_plan_from_extra_args_serial_s1():
    plan = build_plan_from_extra_args(
        {"num_chunks": 1, "chunk_schedule": "serial"},
        denoise_steps=4,
        stages=1,
        layer_groups=1,
        blocks=8,
    )
    assert plan.num_slots > 0


def test_build_plan_from_extra_args_rejects_bad_stages():
    with pytest.raises(ValueError, match="stage_parallel_size"):
        build_plan_from_extra_args(
            {},
            denoise_steps=4,
            stages=3,
            layer_groups=1,
            blocks=8,
        )


def test_build_plan_from_extra_args_vertical_fills_history():
    plan = build_plan_from_extra_args(
        {"num_chunks": 1, "chunk_schedule": "serial"},
        denoise_steps=4,
        stages=5,
        layer_groups=1,
        blocks=8,
        max_history_chunks=2,
    )
    assert plan.num_slots > 0
