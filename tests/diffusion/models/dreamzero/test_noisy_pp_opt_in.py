# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DreamZero Legacy Session vs NoisyKV coexistence (parameter opt-in)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.experimental.ar_diffusion.capability import (
    SupportsARDiffusionChunkPipeline,
    SupportsARDiffusionPipeline,
)
from vllm_omni.experimental.ar_diffusion.chunk_config import (
    resolve_enable_noisy_kv,
    resolve_enable_noisy_pp,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_enable_noisy_kv_defaults_off():
    od = SimpleNamespace(ar_diffusion_stage_config=None, model_config={})
    assert resolve_enable_noisy_kv(od) is False


def test_enable_noisy_kv_from_model_config():
    od = SimpleNamespace(
        ar_diffusion_stage_config=None,
        model_config={"enable_noisy_kv": True},
    )
    assert resolve_enable_noisy_kv(od) is True


def test_enable_noisy_kv_from_stage_config():
    od = SimpleNamespace(
        ar_diffusion_stage_config={"enable_noisy_kv": True},
        model_config={},
    )
    assert resolve_enable_noisy_kv(od) is True


def test_enable_noisy_pp_legacy_alias_still_works():
    od = SimpleNamespace(
        ar_diffusion_stage_config={"enable_noisy_pp": True},
        model_config={},
    )
    assert resolve_enable_noisy_pp(od) is True
    assert resolve_enable_noisy_kv(od) is True


def test_dreamzero_chunk_flag_defaults_false_without_init():
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    assert pipe.ar_diffusion_use_chunk_pipeline is False
    pipe._enable_noisy_kv = True
    assert pipe.ar_diffusion_use_chunk_pipeline is True


def test_runner_chunk_selection_logic_dual_path():
    """Mirror ARDiffusionModelRunner prealloc selection for dual-capability pipes."""

    class Dual:
        ar_diffusion_use_chunk_pipeline = False

        def ar_diffusion_kv_cache_spec(self):
            raise NotImplementedError

        def bind_ar_diffusion_state(self, *a, **k):
            raise NotImplementedError

        def reset_ar_diffusion_session(self, *a, **k):
            raise NotImplementedError

        def close_ar_diffusion_session(self, *a, **k):
            raise NotImplementedError

        def ar_diffusion_noisy_kv_spec(self):
            raise NotImplementedError

        def bind_ar_diffusion_chunk_context(self, *a, **k):
            raise NotImplementedError

    pipe = Dual()
    assert isinstance(pipe, SupportsARDiffusionChunkPipeline)
    assert isinstance(pipe, SupportsARDiffusionPipeline)
    use_chunk = getattr(pipe, "ar_diffusion_use_chunk_pipeline", None)
    if use_chunk is None:
        use_chunk = not isinstance(pipe, SupportsARDiffusionPipeline)
    assert use_chunk is False
    pipe.ar_diffusion_use_chunk_pipeline = True
    use_chunk = getattr(pipe, "ar_diffusion_use_chunk_pipeline", None)
    if use_chunk is None:
        use_chunk = not isinstance(pipe, SupportsARDiffusionPipeline)
    assert use_chunk is True


def test_runner_chunk_selection_logic_chunk_only():
    class ChunkOnly:
        def ar_diffusion_noisy_kv_spec(self):
            raise NotImplementedError

        def bind_ar_diffusion_chunk_context(self, *a, **k):
            raise NotImplementedError

    pipe = ChunkOnly()
    assert isinstance(pipe, SupportsARDiffusionChunkPipeline)
    assert not isinstance(pipe, SupportsARDiffusionPipeline)
    use_chunk = getattr(pipe, "ar_diffusion_use_chunk_pipeline", None)
    if use_chunk is None:
        use_chunk = not isinstance(pipe, SupportsARDiffusionPipeline)
    assert use_chunk is True
