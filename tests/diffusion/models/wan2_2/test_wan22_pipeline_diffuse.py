# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import run_chunk_pipeline
from vllm_omni.diffusion.media import VideoTensorEncoding, VideoTensorLayout, VideoValueRange
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanSelfAttention
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StubTransformer(nn.Module):
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _StubTextEncoder(nn.Module):
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _StubVaeConfig:
    latents_mean = [0.0, 0.0, 0.0, 0.0]
    latents_std = [1.0, 1.0, 1.0, 1.0]
    z_dim = 4


class _StubVae(nn.Module):
    dtype = torch.float32
    config = _StubVaeConfig()

    def decode(self, latents, return_dict=False):
        del return_dict
        batch, _, frames, height, width = latents.shape
        return (torch.zeros(batch, 3, frames, height, width),)


class _StubScheduler:
    def __init__(self, timesteps: list[int]) -> None:
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))


@contextmanager
def _noop_progress_bar(*args, **kwargs):
    del args, kwargs

    class _Bar:
        def update(self) -> None:
            return None

    yield _Bar()


def _stub_encode_prompt(
    prompt,
    negative_prompt=None,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    max_sequence_length=512,
    device=None,
    dtype=None,
):
    del negative_prompt, do_classifier_free_guidance, device, dtype
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    n = batch_size * num_videos_per_prompt
    hidden_size = 8
    prompt_embeds = torch.zeros(n, max_sequence_length, hidden_size)
    return prompt_embeds, None


def _make_pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _StubTransformer()
    pipeline.transformer_2 = None
    pipeline.text_encoder = _StubTextEncoder()
    pipeline.vae = _StubVae()
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=4, out_channels=4)
    pipeline.scheduler = _StubScheduler([9, 5])
    pipeline.od_config = SimpleNamespace(
        flow_shift=5.0,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.boundary_ratio = 0.875
    pipeline.expand_timesteps = False
    pipeline.is_dmd = False
    pipeline.chunk_pipeline_mode = False
    pipeline.is_causalwan_dmd = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = _stub_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = lambda **kwargs: torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
    pipeline.progress_bar = _noop_progress_bar
    return pipeline


def _make_sampling(**overrides):
    values: dict[str, object] = {
        "height": None,
        "width": None,
        "num_frames": 1,
        "num_inference_steps": 2,
        "guidance_scale_provided": True,
        "guidance_scale": 1.0,
        "guidance_scale_2": None,
        "guidance_scale_2_provided": False,
        "boundary_ratio": None,
        "generator": None,
        "seed": None,
        "num_outputs_per_prompt": 1,
        "max_sequence_length": 32,
        "latents": None,
        "output_type": "latent",
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("sampling_params_kwargs", "expected_low", "expected_high"),
    [
        ({}, 4.0, 4.0),
        ({"guidance_scale": 0.0}, 0.0, 0.0),
        ({"guidance_scale": 1.0}, 1.0, 1.0),
        ({"guidance_scale": 3.0, "guidance_scale_2": 5.0}, 3.0, 5.0),
    ],
)
def test_forward_delegates_denoising_to_diffuse(
    sampling_params_kwargs: dict[str, float],
    expected_low: float,
    expected_high: float,
) -> None:
    pipeline = _make_pipeline()
    captured: dict[str, object] = {}

    def _fake_diffuse(**kwargs):
        captured.update(kwargs)
        return kwargs["latents"] + 1

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]

    mock_req = OmniDiffusionRequest(
        prompt="prompt",
        request_id="test-req",
        sampling_params=OmniDiffusionSamplingParams(
            num_frames=1,
            num_inference_steps=2,
            max_sequence_length=32,
            output_type="latent",
            **sampling_params_kwargs,
        ),
    )
    batch = DiffusionRequestBatch(requests=[mock_req])

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    assert torch.equal(outputs[0].output, torch.ones((1, 4, 1, 8, 8)))
    assert torch.equal(captured["prompt_embeds"], torch.zeros(1, 32, 8))
    assert torch.equal(captured["timesteps"], pipeline.scheduler.timesteps)
    assert captured["guidance_low"] == expected_low
    assert captured["guidance_high"] == expected_high
    assert captured["boundary_timestep"] == pytest.approx(875.0)
    assert captured["latent_condition"] is None
    assert captured["first_frame_mask"] is None
    assert pipeline.scheduler.set_timesteps_calls == [(2, torch.device("cpu"))]


def test_forward_batches_text_generators_latents_and_splits_outputs() -> None:
    pipeline = _make_pipeline()
    encode_call = {}
    prepare_call = {}

    def _fake_encode_prompt(**kwargs):
        encode_call.update(kwargs)
        batch_size = len(kwargs["prompt"])
        n = batch_size * kwargs["num_videos_per_prompt"]
        return torch.arange(n, dtype=torch.float32).view(n, 1, 1), torch.zeros(n, 1, 1)

    def _fake_prepare_latents(**kwargs):
        prepare_call.update(kwargs)
        return kwargs["latents"]

    pipeline.encode_prompt = _fake_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = _fake_prepare_latents  # type: ignore[method-assign]
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]  # type: ignore[method-assign]

    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)
    latents_a = torch.zeros(2, 4, 1, 2, 2)
    latents_b = torch.ones(2, 4, 1, 2, 2)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first", "negative_prompt": "bad first"},
                sampling_params=_make_sampling(
                    generator=gen_a,
                    latents=latents_a,
                    num_outputs_per_prompt=2,
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "negative_prompt": "bad second"},
                sampling_params=_make_sampling(
                    generator=gen_b,
                    latents=latents_b,
                    num_outputs_per_prompt=2,
                ),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    assert encode_call["prompt"] == ["first", "second"]
    assert encode_call["negative_prompt"] == ["bad first", "bad second"]
    assert prepare_call["batch_size"] == 4
    assert prepare_call["generator"] == [gen_a, gen_a, gen_b, gen_b]
    torch.testing.assert_close(prepare_call["latents"], torch.cat([latents_a, latents_b]))
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output, latents_a)
    torch.testing.assert_close(outputs[1].output, latents_b)


def test_forward_emits_request_local_typed_media_after_vae_decode() -> None:
    pipeline = _make_pipeline()

    def _fake_diffuse(
        *,
        latents,
        timesteps,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_low,
        guidance_high,
        boundary_timestep,
        dtype,
        attention_kwargs,
        latent_condition,
        first_frame_mask,
        generator,
    ):
        del (
            timesteps,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_low,
            guidance_high,
            boundary_timestep,
            dtype,
            attention_kwargs,
            latent_condition,
            first_frame_mask,
            generator,
        )
        return torch.zeros_like(latents)

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="prompt",
                request_id="request-0",
                sampling_params=OmniDiffusionSamplingParams(
                    num_frames=1,
                    num_inference_steps=2,
                    max_sequence_length=32,
                    output_type="np",
                ),
            )
        ]
    )

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    assert outputs[0].output is None
    assert outputs[0].media is not None
    assert outputs[0].media.prepared_for_transport is False
    assert outputs[0].media.video.tensor.shape == (1, 3, 1, 8, 8)
    assert outputs[0].media.video.spec.layout is VideoTensorLayout.BCTHW
    assert outputs[0].media.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
    assert outputs[0].media.video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE


@pytest.mark.parametrize(
    ("output_type", "expected_empty_cache_calls"),
    [("latent", 0), ("np", 1)],
)
def test_forward_only_clears_cache_before_vae_decode(
    monkeypatch: pytest.MonkeyPatch,
    output_type: str,
    expected_empty_cache_calls: int,
) -> None:
    pipeline = _make_pipeline()
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]  # type: ignore[method-assign]
    empty_cache_calls: list[None] = []
    platform = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: empty_cache_calls.append(None),
    )
    module = importlib.import_module(Wan22Pipeline.__module__)
    monkeypatch.setattr(module, "current_omni_platform", platform)
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="prompt",
                request_id="request-0",
                sampling_params=OmniDiffusionSamplingParams(
                    num_frames=1,
                    num_inference_steps=2,
                    max_sequence_length=32,
                    output_type=output_type,
                ),
            )
        ]
    )

    pipeline.forward(batch)

    assert len(empty_cache_calls) == expected_empty_cache_calls


@pytest.mark.parametrize("decoded", [None, torch.empty(0)], ids=["none", "empty-tensor"])
def test_forward_keeps_legacy_output_on_non_owner_vae_rank(decoded: torch.Tensor | None) -> None:
    # Non-owner VAE ranks can return None or an empty tensor. Exercise the full
    # forward path through typed-media dispatch and per-request output splitting;
    # neither placeholder may be wrapped as video media or require a tensor shape.
    pipeline = _make_pipeline()
    pipeline.vae.decode = lambda latents, return_dict=False: (decoded,)  # type: ignore[assignment]
    pipeline.diffuse = lambda **kwargs: torch.zeros_like(kwargs["latents"])  # type: ignore[method-assign]

    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="prompt",
                request_id="request-0",
                sampling_params=OmniDiffusionSamplingParams(
                    num_frames=1,
                    num_inference_steps=2,
                    max_sequence_length=32,
                    output_type="np",
                ),
            )
        ]
    )

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    assert outputs[0].media is None
    if decoded is None:
        assert outputs[0].output is None
    else:
        assert outputs[0].output is not None
        assert outputs[0].output.numel() == 0


def test_forward_batches_precomputed_prompt_embeddings() -> None:
    pipeline = _make_pipeline()
    diffuse_call = {}
    pipeline.encode_prompt = lambda **kwargs: pytest.fail("text encoder must not run")  # type: ignore[method-assign]
    pipeline.prepare_latents = lambda **kwargs: torch.zeros(kwargs["batch_size"], 4, 1, 2, 2)  # type: ignore[method-assign]

    def _fake_diffuse(**kwargs):
        diffuse_call.update(kwargs)
        return kwargs["latents"]

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]
    embeds_a = torch.zeros(3, 4)
    embeds_b = torch.ones(3, 4)
    negative_a = torch.full((3, 4), 2.0)
    negative_b = torch.full((3, 4), 3.0)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt_embeds": embeds_a, "negative_prompt_embeds": negative_a},
                sampling_params=_make_sampling(guidance_scale=4.0),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt_embeds": embeds_b, "negative_prompt_embeds": negative_b},
                sampling_params=_make_sampling(guidance_scale=4.0),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    torch.testing.assert_close(diffuse_call["prompt_embeds"], torch.stack([embeds_a, embeds_b]))
    torch.testing.assert_close(diffuse_call["negative_prompt_embeds"], torch.stack([negative_a, negative_b]))
    assert len(outputs) == 2


def test_prepare_latents_with_request_generators_matches_single_generation() -> None:
    pipeline = _make_pipeline()
    kwargs = {
        "num_channels_latents": 4,
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }

    batched = Wan22Pipeline.prepare_latents(
        pipeline,
        batch_size=2,
        generator=[torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)],
        **kwargs,
    )
    singles = torch.cat(
        [
            Wan22Pipeline.prepare_latents(
                pipeline,
                batch_size=1,
                generator=torch.Generator().manual_seed(seed),
                **kwargs,
            )
            for seed in (1, 2)
        ]
    )

    torch.testing.assert_close(batched, singles)
    assert not torch.equal(batched[0], batched[1])


def test_diffuse_runs_prediction_and_scheduler_for_each_timestep() -> None:
    pipeline = _make_pipeline()
    latents = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    timesteps = torch.tensor([7, 3], dtype=torch.int64)
    prompt_embeds = torch.randn(1, 8)

    predict_calls: list[dict[str, object]] = []
    scheduler_calls: list[tuple[float, int, float, bool]] = []

    def _fake_predict_noise_maybe_with_cfg(**kwargs):
        predict_calls.append(kwargs)
        timestep = kwargs["positive_kwargs"]["timestep"]
        assert isinstance(timestep, torch.Tensor)
        return torch.full_like(latents, float(timestep[0].item()))

    def _fake_scheduler_step_maybe_with_cfg(noise_pred, t, current_latents, do_true_cfg):
        scheduler_calls.append(
            (float(noise_pred[0, 0, 0, 0, 0]), int(t.item()), float(current_latents.sum()), do_true_cfg)
        )
        return current_latents + noise_pred

    pipeline.predict_noise_maybe_with_cfg = _fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = _fake_scheduler_step_maybe_with_cfg  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=2.0,
        boundary_timestep=5.0,
        dtype=torch.float32,
        attention_kwargs={},
    )

    assert len(predict_calls) == 2
    assert predict_calls[0]["true_cfg_scale"] == 1.0
    assert predict_calls[1]["true_cfg_scale"] == 2.0
    assert scheduler_calls == [
        (7.0, 7, 0.0, False),
        (3.0, 3, 28.0, False),
    ]
    assert torch.equal(result, torch.full_like(latents, 10.0))


def test_diffuse_publishes_precomputed_forward_context_timesteps(monkeypatch) -> None:
    """Native PP avoids a per-step accelerator scalar readback too."""
    pipeline = _make_pipeline()
    timesteps = torch.tensor([[7], [3]], dtype=torch.int64)
    recorded: list[tuple[int, object, float | None]] = []

    def record(step_idx, timestep=None, scheduler=None, normalized_timestep=None, total_steps=None):
        del scheduler, total_steps
        recorded.append((step_idx, timestep, normalized_timestep))

    pipeline.record_denoise_step = record  # type: ignore[method-assign]
    pipeline.predict_noise_maybe_with_cfg = (  # type: ignore[method-assign]
        lambda **kwargs: torch.zeros_like(kwargs["positive_kwargs"]["hidden_states"])
    )
    pipeline.scheduler_step_maybe_with_cfg = (  # type: ignore[method-assign]
        lambda noise_pred, t, current_latents, do_true_cfg: current_latents
    )
    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.is_forward_context_available", lambda: True)

    pipeline.diffuse(
        latents=torch.zeros((1, 1, 1, 2, 2)),
        timesteps=timesteps,
        prompt_embeds=torch.zeros(1, 8),
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=1.0,
        boundary_timestep=None,
        dtype=torch.float32,
        attention_kwargs={},
    )

    assert [(step_idx, timestep) for step_idx, timestep, _ in recorded] == [(0, None), (1, None)]
    assert [normalized for _, _, normalized in recorded] == pytest.approx([0.007, 0.003])


class _StubDMDScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.predict_clean_calls: list[tuple[float, float, float]] = []
        self.add_noise_calls: list[tuple[float, float, float]] = []

    def predict_clean(self, model_output, sample, timestep):
        self.predict_clean_calls.append((float(model_output.mean()), float(sample.mean()), float(timestep)))
        return sample - model_output

    def add_noise(self, clean_sample, noise, timestep):
        self.add_noise_calls.append((float(clean_sample.mean()), float(noise.mean()), float(timestep)))
        return clean_sample + 10.0


@pytest.mark.xfail(reason="native DMD numeric parity deferred to the #2/#3 contract audit", strict=True)
def test_diffuse_dmd_updates_sample_through_shared_contract(monkeypatch) -> None:
    """Native DMD steps go through the shared update_sample contract.

    With sigmas [1.0, 0.5, 0.1] at timesteps [1000, 757, 522], constant
    v-prediction 1.0 and constant noise 2.0, the flow updates are:
    step0: x0 = 0 - 1.0 = -1, x' = 0.5*(-1) + 0.5*2 = 0.5
    step1: x0 = 0.5 - 0.5 = 0,  x' = 0.9*0 + 0.1*2 = 0.2
    step2 (final): x0 = 0.2 - 0.1 = 0.1
    """
    monkeypatch.setattr("vllm_omni.diffusion.distributed.pipeline_parallel.get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _make_pipeline()
    pipeline.is_dmd = True
    # Single-expert configuration: no boundary, every step renoises low.
    pipeline.boundary_ratio = None
    scheduler = _StubDMDScheduler()
    scheduler.timesteps = torch.tensor([1000.0, 757.0, 522.0])
    scheduler.sigmas = torch.tensor([1.0, 0.5, 0.1])
    pipeline.scheduler = scheduler
    latents = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
    timesteps = torch.tensor([1000.0, 757.0, 522.0])

    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: torch.ones_like(latents)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.randn_tensor",
        lambda *args, **kwargs: torch.full(args[0], 2.0, dtype=kwargs["dtype"]),
    )

    result = pipeline.diffuse(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=torch.zeros(1, 8),
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=1.0,
        boundary_timestep=None,
        dtype=torch.float32,
        attention_kwargs={},
        generator=torch.Generator(device="cpu").manual_seed(1),
    )

    torch.testing.assert_close(result, torch.tensor([[[[[0.1]]]]]))


def test_chunk_pipeline_publishes_precomputed_denoise_timesteps(monkeypatch) -> None:
    """Chunk PP must not turn the per-slot CUDA timestep into a Python scalar."""
    pipeline = _make_pipeline()
    pipeline.transformer.start_layer = 0
    pipeline.transformer.end_layer = 1
    # A scheduler may expose [steps, 1] rather than a flat tensor.  The
    # chunk implementation accepts either form and must still avoid a scalar
    # readback inside each slot.
    timesteps = torch.tensor([[900.0], [500.0], [100.0]])
    latents = torch.zeros((1, 4, 2, 1, 1), dtype=torch.float32)
    recorded: list[tuple[int, object, float | None]] = []

    def record(step_idx, timestep=None, scheduler=None, normalized_timestep=None, total_steps=None):
        del scheduler, total_steps
        recorded.append((step_idx, timestep, normalized_timestep))

    def fake_predict_noise(**kwargs):
        return torch.zeros_like(kwargs["hidden_states"])

    def fake_run_chunk_pipeline(*args, **kwargs):
        model_input = kwargs["initial_latents"]
        for step_idx in range(4):
            timestep = kwargs["timesteps"][step_idx : step_idx + 1] if step_idx < 3 else timesteps.new_zeros(1)
            kwargs["predict_noise"](model_input, timestep, 0, step_idx, None)
        return kwargs["initial_latents"], {}

    pipeline.predict_noise = fake_predict_noise  # type: ignore[method-assign]
    pipeline.record_denoise_step = record  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.run_chunk_pipeline",
        fake_run_chunk_pipeline,
    )
    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.is_forward_context_available", lambda: True)
    req = SimpleNamespace(
        num_reqs=1,
        sampling_params_list=[
            SimpleNamespace(
                extra_args={"chunk_frames": 5, "chunk_conditioning": "latest_kv", "kv_history_chunks": 1},
                num_outputs_per_prompt=1,
                seed=1,
            )
        ],
    )

    result = pipeline._diffuse_chunks(
        req,
        latents,
        timesteps,
        torch.zeros(1, 2, 8),
        torch.float32,
        torch.Generator(device="cpu").manual_seed(1),
    )

    assert result is latents
    assert [(step_idx, timestep) for step_idx, timestep, _ in recorded] == [
        (0, None),
        (1, None),
        (2, None),
        (3, None),
    ]
    assert [normalized for _, _, normalized in recorded] == pytest.approx([0.9, 0.5, 0.1, 0.0])


def test_select_dit_routes_steps_by_boundary_timestep() -> None:
    """CausalWan DMD routes each step to the tower its timestep belongs to."""
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import CAUSALWAN_DMD_TIMESTEPS

    pipeline = _make_pipeline()
    high, low = _StubTransformer(), _StubTransformer()
    pipeline.transformer, pipeline.transformer_2 = high, low
    pipeline.boundary_ratio = 0.875
    # The stub scheduler already exposes num_train_timesteps=1000.
    assert [pipeline._select_dit(torch.tensor(t)) for t in CAUSALWAN_DMD_TIMESTEPS] == [
        high,
        low,
        low,
        low,
        low,
        low,
        low,
        low,
    ]
    # The clean pass (t=0) also runs on the low-noise tower.
    assert pipeline._select_dit(torch.tensor(0.0)) is low
    assert pipeline._select_dit(1000.0) is high


def test_select_dit_falls_back_to_single_tower() -> None:
    """FastWan's single-tower checkpoint keeps routing to its only transformer."""
    pipeline = _make_pipeline()
    only = _StubTransformer()
    pipeline.transformer, pipeline.transformer_2 = only, None
    pipeline.boundary_ratio = 0.875
    assert pipeline._select_dit(torch.tensor(100.0)) is only
    assert pipeline._select_dit(torch.tensor(1000.0)) is only


def test_dit_router_precomputes_per_step_towers() -> None:
    """The router materializes every step's tower once, clean pass included.

    The chunk pipeline uses this table instead of reading a CUDA scalar per
    slot, so the choice must match _select_dit for every denoise step and
    route the synthetic t=0 clean pass to the low-noise tower.
    """
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import CAUSALWAN_DMD_TIMESTEPS

    pipeline = _make_pipeline()
    high, low = _StubTransformer(), _StubTransformer()
    pipeline.transformer, pipeline.transformer_2 = high, low
    pipeline.boundary_ratio = 0.875
    timesteps = torch.tensor(CAUSALWAN_DMD_TIMESTEPS)

    router = pipeline._dit_router(timesteps, pipeline._request_boundary_timestep())
    assert len(router) == len(CAUSALWAN_DMD_TIMESTEPS) + 1
    assert router[: len(CAUSALWAN_DMD_TIMESTEPS)] == [high] + [low] * (len(CAUSALWAN_DMD_TIMESTEPS) - 1)
    assert router[-1] is low

    # A request-level boundary overrides the engine default.
    request_router = pipeline._dit_router(timesteps, boundary_timestep=900.0)
    assert request_router[0] is high  # 1000 >= 900
    assert request_router[1] is low  # 850 < 900
    assert request_router[2] is low  # 700 < 900

    # Single-tower checkpoints collapse to their only transformer.
    pipeline.transformer_2 = None
    single = pipeline._dit_router(timesteps, pipeline._request_boundary_timestep())
    assert all(model is high for model in single)


def test_diffuse_chunks_accepts_eight_step_causalwan_schedule(monkeypatch) -> None:
    """The CausalWan 8-step DMD schedule runs through the chunk pipeline."""
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import CAUSALWAN_DMD_TIMESTEPS

    pipeline = _make_pipeline()
    high, low = _StubTransformer(), _StubTransformer()
    high.start_layer, high.end_layer = 0, 1
    low.start_layer, low.end_layer = 0, 1
    pipeline.transformer, pipeline.transformer_2 = high, low
    pipeline.boundary_ratio = 0.875
    timesteps = torch.tensor(CAUSALWAN_DMD_TIMESTEPS)
    latents = torch.zeros((1, 4, 2, 1, 1), dtype=torch.float32)
    routed: list[tuple[int, object]] = []
    towers: list[object] = []
    step_noises_used: list[object] = []

    def fake_run_chunk_pipeline(**kwargs):
        assert len(kwargs["timesteps"]) == 8
        # One shared re-noising tensor per transition (8 steps → 7).
        assert len(kwargs["step_noises"]) == 7
        step_noises_used.append([t.shape for t in kwargs["step_noises"]])
        model_input = kwargs["initial_latents"]
        for step_idx in range(9):
            timestep = kwargs["timesteps"][step_idx : step_idx + 1] if step_idx < 8 else timesteps.new_zeros(1)
            kwargs["predict_noise"](model_input, timestep, 0, step_idx)
        return kwargs["initial_latents"], {}

    def fake_predict_noise(**kwargs):
        towers.append(kwargs["current_model"])
        return torch.zeros_like(kwargs["hidden_states"])

    def record(step_idx, timestep=None, normalized_timestep=None, **kwargs):
        del kwargs
        routed.append((step_idx, normalized_timestep))

    pipeline.predict_noise = fake_predict_noise  # type: ignore[method-assign]
    pipeline.record_denoise_step = record  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.run_chunk_pipeline",
        fake_run_chunk_pipeline,
    )
    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.is_forward_context_available", lambda: True)
    req = SimpleNamespace(
        num_reqs=1,
        sampling_params_list=[
            SimpleNamespace(
                extra_args={"chunk_frames": 5, "chunk_conditioning": "latest_kv", "kv_history_chunks": 1},
                num_outputs_per_prompt=1,
                seed=1,
            )
        ],
    )

    result = pipeline._diffuse_chunks(
        req,
        latents,
        timesteps,
        torch.zeros(1, 2, 8),
        torch.float32,
        torch.Generator(device="cpu").manual_seed(1),
    )

    assert result is latents
    # 8 denoise steps route their towers through _select_dit (step 0 → high,
    # the rest → low); the t=0 clean pass is synthetic and does not consume a
    # re-noising tensor.
    assert len(step_noises_used[0]) == 7
    assert towers == [high] + [low] * 8
    assert [(step_idx, normalized) for step_idx, normalized in routed][-1] == (8, 0.0)
    assert [normalized for _, normalized in routed] == pytest.approx(
        [t / 1000.0 for t in CAUSALWAN_DMD_TIMESTEPS] + [0.0]
    )


def _make_gate_loading_pipeline():
    pipeline = Wan22Pipeline.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    gate = WanSelfAttention.__new__(WanSelfAttention)
    nn.Module.__init__(gate)
    gate.to_gate_compress = nn.Linear(1, 1)
    pipeline.gate_holder = gate
    return pipeline, gate


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("pipeline_wan2_2", "Wan22Pipeline"),
        ("pipeline_wan2_2_i2v", "Wan22I2VPipeline"),
        ("pipeline_wan2_2_s2v", "Wan22S2VPipeline"),
        ("pipeline_wan2_2_vace", "Wan22VACEPipeline"),
    ],
)
def test_wan_pipeline_loaders_share_optional_gate_cleanup(monkeypatch, module_name, class_name) -> None:
    module = importlib.import_module(f"vllm_omni.diffusion.models.wan2_2.{module_name}")
    pipeline_cls = getattr(module, class_name)
    pipeline = pipeline_cls.__new__(pipeline_cls)
    expected = {"loaded"}

    def fake_loader(model, weights):
        assert model is pipeline
        assert list(weights) == [("weight", torch.ones(1))]
        return expected

    monkeypatch.setattr(module, "load_wan_weights_with_optional_gate", fake_loader)

    assert pipeline_cls.load_weights(pipeline, iter((("weight", torch.ones(1)),))) is expected


def test_load_weights_removes_unloaded_vsa_gate(monkeypatch) -> None:
    pipeline, gate = _make_gate_loading_pipeline()

    class _Loader:
        def __init__(self, model):
            del model

        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.AutoWeightsLoader", _Loader)
    pipeline.load_weights(iter((("other.weight", torch.ones(1)),)))

    assert pipeline.has_gate_compress_weights is False
    assert gate.to_gate_compress is None


def test_load_weights_keeps_trained_vsa_gate(monkeypatch) -> None:
    pipeline, gate = _make_gate_loading_pipeline()
    original_gate = gate.to_gate_compress

    class _Loader:
        def __init__(self, model):
            del model

        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.AutoWeightsLoader", _Loader)
    pipeline.load_weights(iter((("gate_holder.to_gate_compress.weight", torch.ones(1)),)))

    assert pipeline.has_gate_compress_weights is True
    assert gate.to_gate_compress is original_gate


class _StubChunkScheduler:
    def predict_clean(self, model_output, sample, timestep):
        del timestep
        return sample - model_output

    def add_noise(self, clean_sample, noise, timestep):
        del timestep
        return clean_sample + noise


class _FakeCudaStream:
    pass


class _FakeCudaEvent:
    def __init__(self, enable_timing=False):
        del enable_timing

    def record(self, stream=None):
        del stream

    def elapsed_time(self, other):
        del other
        return 0.0


def _patch_cpu_chunk_pp_runtime(monkeypatch, rank_in_group=0, world_size=1):
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.chunk_pipeline_parallel.get_pp_group",
        lambda: SimpleNamespace(rank_in_group=rank_in_group, world_size=world_size, device_group=None, cpu_group=None),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeCudaStream())
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda *a, **k: 0)


def test_run_chunk_pipeline_accepts_steps_by_1_timesteps(monkeypatch) -> None:
    """A [steps, 1] scheduler timesteps tensor must not turn into a float([]) readback."""
    _patch_cpu_chunk_pp_runtime(monkeypatch)
    scheduler = _StubChunkScheduler()
    timesteps = torch.tensor([[900.0], [500.0], [100.0]])
    chunks = 6
    shape = (1, 4, 2, 1, 1)

    def fake_predict_noise(
        model_input, timestep, temporal_offset, step_idx, producer=None, intermediate_tensors=None, kv_context=None
    ):
        del timestep, temporal_offset, step_idx, producer, intermediate_tensors, kv_context
        return torch.zeros_like(model_input)

    result, metrics = run_chunk_pipeline(
        predict_noise=fake_predict_noise,
        update_sample=lambda **kwargs: kwargs.get("sample"),
        timesteps=timesteps,
        shape=shape,
        chunks=chunks,
        seed=0,
        device=torch.device("cpu"),
    )

    assert result.shape == (1, 4, chunks * shape[2], 1, 1)
    steps = metrics["ranks"][0]["steps"]
    assert len(steps) == chunks * 3
    by_step: dict[int, list[float]] = {}
    for record in steps:
        by_step.setdefault(record["step_idx"], []).append(record["timestep"])
    assert by_step[0] == [900.0] * chunks
    assert by_step[1] == [500.0] * chunks
    assert by_step[2] == [100.0] * chunks
