# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(prompt, num_inference_steps):
    req = MagicMock()
    req.prompts = [prompt]
    req.sampling_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
    )
    return req


def test_forward_rejects_empty_prompt_and_zero_steps():
    pipeline = object.__new__(Flux2Pipeline)
    pipeline.vae_scale_factor = 8

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        pipeline.forward(_request("", 4))

    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        pipeline.forward(_request("valid prompt", 0))
