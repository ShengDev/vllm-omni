# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WaveServe model unit checks (layer split + FlowEuler + latent adapter).

Omni / diffusion registry wiring lives in
``tests/entrypoints/test_resolve_waveserve_wan_config.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.waveserve_wan.pipeline_waveserve_wan import (
    FlowEuler,
    WaveServeWanPipeline,
    _LatentChunkAdapter,
)
from vllm_omni.diffusion.models.waveserve_wan.transformer import StageWanTransformer, stage_layer_range
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.chunk_executor import ARDiffusionChunkContext, ChunkRunSpec, ChunkTopology
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage_layer_range_covers_all_layers():
    covered: list[int] = []
    for g in range(2):
        start, end = stage_layer_range(30, g, 2)
        covered.extend(range(start, end))
    assert covered == list(range(30))
    assert stage_layer_range(30, 0, 2) == (0, 15)
    assert stage_layer_range(30, 1, 2) == (15, 30)


def test_flow_euler_matches_diffusers_shifted_endpoint():
    """Match FlowMatchEulerDiscreteScheduler: linspace ends at warp(1/1000), then warp again."""
    shift = 5.0
    steps = 4

    def warp(sigma: torch.Tensor) -> torch.Tensor:
        return shift * sigma / (1 + (shift - 1) * sigma)

    sampler = FlowEuler(steps, shift=shift)
    expected = warp(torch.linspace(1.0, float(warp(torch.tensor(1.0 / 1000))), steps, dtype=torch.float64))
    assert sampler.sigmas == [*expected.float().tolist(), 0.0]
    # Diffusers' own endpoint double-warp yields a larger last step than warp(1/1000).
    last_delta = abs(sampler.sigmas[-1] - sampler.sigmas[-2])
    assert last_delta > 0.01


def test_waveserve_chunk_noise_matches_reference_seed():
    shape = (1, 16, 1, 2, 2)
    seed, chunk = 7, 2
    adapter = _LatentChunkAdapter(
        SimpleNamespace(prepare_cell=lambda *_a, **_k: None),  # type: ignore[arg-type]
        sampler=FlowEuler(3, shift=5.0),
        prompt_embeds=torch.empty(1, 1, 32),
        latent_shape=shape,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    reference = torch.randn(shape, generator=torch.Generator().manual_seed(seed * 1_000_003 + chunk * 4096))
    torch.testing.assert_close(adapter._init_noise(chunk), reference)
    torch.testing.assert_close(adapter._init_noise(chunk), reference)


def test_chunk_schedule_rejects_unknown_values():
    pipeline = WaveServeWanPipeline.__new__(WaveServeWanPipeline)
    pipeline.stage_parallel_size = 1
    pipeline.layer_groups = 1
    pipeline.max_history_chunks = 1
    pipeline.transformer = SimpleNamespace(num_layers=1, local_num_layers=1)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="sched-0",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"chunk_schedule": "unsupported"}),
    )
    with pytest.raises(ValueError, match="chunk_schedule"):
        pipeline._plan_for(req)


@pytest.mark.parametrize("source", ["num_inference_steps", "num_denoise_steps", "sampling_params"])
@pytest.mark.parametrize("steps", [0, -1])
def test_chunk_schedule_rejects_nonpositive_steps(source, steps):
    pipeline = WaveServeWanPipeline.__new__(WaveServeWanPipeline)
    pipeline.stage_parallel_size = 1
    pipeline.layer_groups = 1
    pipeline.max_history_chunks = 1
    pipeline.transformer = SimpleNamespace(num_layers=1, local_num_layers=1)
    extra = {} if source == "sampling_params" else {source: steps}
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="invalid-steps",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=steps if source == "sampling_params" else 4, extra_args=extra
        ),
    )
    with pytest.raises(ValueError, match="num_denoise_steps must be positive"):
        pipeline._plan_for(req)


@pytest.mark.parametrize(("stages", "expected_steps"), [(1, 4), (4, 3), (5, 4)])
def test_chunk_schedule_unset_steps_keep_topology_default(stages, expected_steps):
    pipeline = WaveServeWanPipeline.__new__(WaveServeWanPipeline)
    pipeline.stage_parallel_size = stages
    pipeline.layer_groups = 1
    pipeline.max_history_chunks = 6
    pipeline.transformer = SimpleNamespace(num_layers=1, local_num_layers=1)
    req = OmniDiffusionRequest(prompt="x", request_id="default-steps", sampling_params=OmniDiffusionSamplingParams())
    plan, _, steps = pipeline._plan_for(req)
    assert steps == plan.schedule.num_denoise_steps == expected_steps


@pytest.mark.parametrize("shape", [(1, 16, 1, 4, 8), (1, 16, 1, 8, 16)])
def test_pipeline_rejects_token_count_change_before_admission(shape):
    pipeline = WaveServeWanPipeline.__new__(WaveServeWanPipeline)
    torch.nn.Module.__init__(pipeline)
    transformer = StageWanTransformer.__new__(StageWanTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.patch_size = (1, 2, 2)
    transformer.register_parameter("test_device", torch.nn.Parameter(torch.zeros(1)))
    pipeline.transformer = transformer
    pipeline.stage_parallel_size = pipeline.layer_groups = pipeline.max_history_chunks = 1
    pipeline.block_size = pipeline.max_chunk_tokens = 16
    pipeline._chunk_ctx = None
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(1, 1, 1, 16, 16, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    ctx = ARDiffusionChunkContext(ChunkRunSpec(ChunkTopology(1, 1)), NoisyKVState(cache))
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="different-shape",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"latent_shape": shape, "num_denoise_steps": 1}),
    )
    with pipeline.bind_ar_diffusion_chunk_context(ctx):
        with pytest.raises(ValueError, match="does not match preallocated NoisyKV"):
            pipeline.forward(req)
    assert pipeline.block_size == pipeline.max_chunk_tokens == cache.spec.max_chunk_tokens == 16
    assert not ctx.pending and not ctx.inflight and not cache.pool.keys


def test_latent_adapter_g_gt1_packs_hidden_and_advances_on_last_only():
    """Non-last groups forward tokens; only stage-last runs FlowEuler."""
    shape = (1, 4, 1, 2, 2)
    device = torch.device("cpu")
    dtype = torch.float32
    prompt = torch.zeros(1, 8, 16, device=device, dtype=dtype)

    class _FakeTransformer:
        def __init__(self, *, is_stage_first: bool, is_stage_last: bool) -> None:
            self.is_stage_first = is_stage_first
            self.is_stage_last = is_stage_last
            self.start_layer = 0
            self.end_layer = 1
            self.dim = 8

        def prepare_cell(self, *_a, **_k) -> None:
            return None

        def seq_len_for_latent(self, _shape) -> int:
            return 3

        def embed(self, latent):
            return torch.ones(latent.shape[0], 3, 8, device=latent.device, dtype=latent.dtype)

        def block(self, _index, tokens, **_kw):
            return tokens

        def unembed(self, _tokens, *, latent, **_kw):
            return torch.zeros_like(latent)

    sampler = FlowEuler(2, shift=1.0)
    mid = _LatentChunkAdapter(
        _FakeTransformer(is_stage_first=True, is_stage_last=False),  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt,
        latent_shape=shape,
        seed=0,
        device=device,
        dtype=dtype,
    )
    last = _LatentChunkAdapter(
        _FakeTransformer(is_stage_first=False, is_stage_last=True),  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt,
        latent_shape=shape,
        seed=0,
        device=device,
        dtype=dtype,
    )

    tasks = [("r0", (0, 0, 0))]
    mid_out = mid.forward(tasks, [None], hidden=None)
    assert {"latent", "hidden_states"} <= set(mid_out)
    packed = mid.pack_activation(mid_out)
    assert "latent" in packed and "hidden_states" in packed

    last_in = last.unpack_activation(packed)
    last_out = last.forward(tasks, [None], hidden=last_in)
    assert "latent" in last_out
    assert 0 not in last.finished.get("r0", {})
    # Next denoise step on a non-first group still needs peer hidden_states.
    last_out2 = last.forward(
        [("r0", (0, 1, 0))],
        [None],
        hidden={"latent": last_out["latent"], "hidden_states": packed["hidden_states"]},
    )
    assert 0 in last.finished["r0"]
    assert last_out2["latent"].shape == shape
