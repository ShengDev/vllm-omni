# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    pipeline = object.__new__(Flux2Pipeline)
    pipeline.vae_scale_factor = 8
    return pipeline


def _check(pipeline, prompt, prompt_embeds=None, num_inference_steps=4):
    pipeline.check_inputs(
        prompt=prompt,
        height=512,
        width=512,
        prompt_embeds=prompt_embeds,
        num_inference_steps=num_inference_steps,
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
    ],
)
def test_rejects_empty_or_whitespace_prompt(prompt):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _check(pipe, prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        ["valid prompt", ""],
        ["valid prompt", "   "],
        ["   "],
    ],
)
def test_rejects_list_with_empty_or_whitespace_element(prompt):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _check(pipe, prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "valid prompt",
        ["valid prompt", "another valid prompt"],
    ],
)
def test_accepts_valid_prompts(prompt):
    pipe = _make_pipeline()
    _check(pipe, prompt)


def test_rejects_none_without_embeds():
    pipe = _make_pipeline()
    with pytest.raises(ValueError):
        _check(pipe, None)


def test_accepts_none_with_prompt_embeds():
    pipe = _make_pipeline()
    _check(pipe, None, prompt_embeds=torch.randn(1, 256, 4096))


class _StopAfterCheckInputsError(Exception):
    pass


def _make_minimal_request(prompt="valid prompt", *, num_inference_steps=None):
    """Build a minimal request that Flux2Pipeline.forward() reads before validation."""
    params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        seed=42,
    )
    req = MagicMock()
    req.sampling_params = params
    req.prompts = [prompt]
    return req


def _capture_num_inference_steps(pipe, steps):
    """Run pipe.forward() and capture the num_inference_steps passed to check_inputs."""
    captured = {}

    def fake_check_inputs(**kwargs):
        captured["num_inference_steps"] = kwargs.get("num_inference_steps")
        raise _StopAfterCheckInputsError

    original = pipe.check_inputs
    pipe.check_inputs = fake_check_inputs
    try:
        req = _make_minimal_request(num_inference_steps=steps)
        pipe.forward(req)
    except _StopAfterCheckInputsError:
        pass
    finally:
        pipe.check_inputs = original
    return captured.get("num_inference_steps")


def test_forward_preserves_zero_not_default():
    """#9575: num_inference_steps=0 must reach validation as 0, not as default 50."""
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, 0)
    assert captured == 0, f"Expected 0, got {captured}"


def test_forward_preserves_negative():
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, -1)
    assert captured == -1


def test_forward_none_uses_default():
    """None means unset — forward() should substitute the pipeline default."""
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, None)
    assert captured == 50


def test_forward_preserves_positive():
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, 9)
    assert captured == 9


@pytest.mark.parametrize("steps", [0, -1])
def test_check_inputs_rejects_non_positive(steps):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        pipe.check_inputs(
            prompt="valid prompt",
            height=512,
            width=512,
            num_inference_steps=steps,
        )


def test_check_inputs_accepts_none_steps():
    pipe = _make_pipeline()
    pipe.check_inputs(prompt="valid prompt", height=512, width=512, num_inference_steps=None)


@pytest.mark.parametrize("steps", [1, 2])
def test_check_inputs_accepts_positive(steps):
    pipe = _make_pipeline()
    pipe.check_inputs(prompt="valid prompt", height=512, width=512, num_inference_steps=steps)


def test_forward_rejects_zero_inference_steps():
    pipe = _make_pipeline()
    req = _make_minimal_request(num_inference_steps=0)
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        pipe.forward(req)


def test_forward_rejects_empty_prompt():
    pipe = _make_pipeline()
    req = _make_minimal_request(prompt="")
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        pipe.forward(req)
