# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from argparse import Namespace

import pytest
from omegaconf import OmegaConf

from benchmarks.noisy_pp.waveserve_serial_vs_latest import (
    _DEFAULT_DEPLOY,
    _expected_slots,
    _resolve_deploy,
    _resolve_topology,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("steps", "groups", "world", "stages"),
    [(1, 1, 1, 1), (4, 1, 1, 1), (4, 2, 2, 1), (1, 1, 2, 2), (4, 2, 10, 5), (4, 1, None, 5)],
)
def test_benchmark_topology_and_deploy_agree(steps, groups, world, stages):
    args = Namespace(
        denoise_steps=steps, gpus_per_stage=groups, world_size=world, history=2, deploy_config=_DEFAULT_DEPLOY
    )
    resolved_world, resolved_stages, resolved_groups = _resolve_topology(args)
    assert resolved_stages == stages
    assert resolved_world == stages * groups
    assert resolved_groups == groups
    deploy = _resolve_deploy(args, resolved_world, resolved_stages)
    try:
        config = OmegaConf.load(deploy).stages[0]
        assert config.parallel_config.pipeline_parallel_size == resolved_world
        assert config.model_config.ar_diffusion_stage_config.stage_parallel_size == stages
        for regime in ("serial", "latest"):
            assert _expected_slots(2, steps, stages, groups, 2, regime) > 0
    finally:
        if deploy != _DEFAULT_DEPLOY:
            deploy.unlink()


@pytest.mark.parametrize(
    ("steps", "groups", "world", "message"),
    [
        (0, 1, 1, "denoise-steps"),
        (-1, 1, 1, "denoise-steps"),
        (1, 0, 1, "gpus-per-stage"),
        (4, 1, 0, "world-size"),
        (4, 2, 3, "world-size"),
    ],
)
def test_benchmark_rejects_invalid_topology_before_model_load(steps, groups, world, message):
    with pytest.raises(SystemExit, match=message):
        _resolve_topology(Namespace(denoise_steps=steps, gpus_per_stage=groups, world_size=world))
