# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import socket

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.fixture(scope="module")
def waveserve_runner():
    from vllm.config.vllm import set_current_vllm_config

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
    from vllm_omni.platforms import current_omni_platform

    model = os.environ.get("VLLM_OMNI_WAVESERVE_TEST_MODEL")
    if not model or not torch.cuda.is_available():
        pytest.skip("requires CUDA and VLLM_OMNI_WAVESERVE_TEST_MODEL pointing to the real RF checkpoint")
    current_omni_platform.set_device(0)
    device = torch.device("cuda", 0)
    od = OmniDiffusionConfig(
        model=model,
        model_class_name="WaveServeWanPipeline",
        dtype=torch.bfloat16,
        num_gpus=1,
        enforce_eager=True,
        max_num_seqs=2,
        model_config={
            "ar_diffusion_stage_config": {"stage_parallel_size": 1, "max_history_chunks": 1, "max_batch_size": 2},
            "ar_diffusion_kv_config": {"warmup_cudagraph": False},
        },
    )
    config = create_diffusion_vllm_config(device, od)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with set_current_vllm_config(config), set_current_diffusion_config(od):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}"
        )
        initialize_model_parallel()
        try:
            runner = ARDiffusionModelRunner(config, od, device)
            runner.load_model()
            yield runner
        finally:
            destroy_model_parallel()
            torch.distributed.destroy_process_group()


def _request(request_id, *, channels=16):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        prompt="a cat walking on grass",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(
            seed=0,
            num_inference_steps=1,
            extra_args={"chunk_schedule": "serial", "num_chunks": 1, "latent_shape": [1, channels, 3, 60, 104]},
        ),
    )


def _execute(runner, requests, *, batch):
    from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData

    if not batch:
        return [runner.execute_model(requests[0])]
    scheduled = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=req.request_id, req=req) for req in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(requests),
        num_waiting_reqs=0,
    )
    return [row.result for row in runner.execute_model_batch(scheduled, runner.od_config).runner_outputs]


def _assert_released(runner):
    cache = runner.noisy_kv_cache
    assert cache is not None
    assert not cache.pool.keys and len(cache.pool.free) == cache.capacity
    assert runner.pipeline._chunk_ctx is None


def test_real_waveserve_runner_preallocates_noisy_kv(waveserve_runner):
    runner = waveserve_runner
    assert runner._ar_diffusion_chunk_capability is runner.pipeline
    assert runner.noisy_kv_cache.spec == runner.pipeline.ar_diffusion_noisy_kv_spec()
    assert runner.noisy_kv_cache.capacity == 8
    assert runner.kv_cache is None and runner._ar_diffusion_capability is None
    _assert_released(runner)


@pytest.mark.parametrize("batch", [False, True])
def test_real_waveserve_runner_success_and_failure_release_kv(waveserve_runner, batch):
    runner = waveserve_runner
    outputs = _execute(runner, [_request("ok-0"), _request("ok-1")] if batch else [_request("ok-0")], batch=batch)
    assert len(outputs) == (2 if batch else 1)
    for output in outputs:
        assert output is not None and torch.isfinite(output.output).all()
    _assert_released(runner)
    # 保持 token 数不变，用真实 patch_embedding 的通道错误触发 KV 分配后的失败。
    before = len(runner._perf_e2e_times)
    with pytest.raises(RuntimeError):
        _execute(runner, [_request("bad", channels=1)], batch=batch)
    assert len(runner._perf_e2e_times) == before
    _assert_released(runner)
    recovered = _execute(runner, [_request("recovered")], batch=batch)
    assert torch.isfinite(recovered[0].output).all()
    _assert_released(runner)
