# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausalWan DMD sample-update contract, aligned with the reference inference.

Reference: FastVideo ``CausalDMDDenosingStage`` (hao-ai-lab/FastVideo) and its
``SelfForcingFlowMatchScheduler``. The public euler scheduler keeps its
upstream behavior; this module adds only the CausalWan-specific update math
as pure functions over sigmas, so the native and chunk-PP paths share one
implementation (review #3) and the dual-tower KV contract (review #2) has an
explicit producer identity.

Sigma/timestep mapping follows the reference: the sigma for a (possibly
distilled) timestep is the nearest entry of the scheduler's sigma table
(``torch.argmin`` over |timesteps - t|).
"""

from __future__ import annotations

import torch

__all__ = [
    "sigma_for_timestep",
    "predict_clean",
    "predict_boundary_state",
    "renoise_low",
    "renoise_high",
    "DmdUpdateKind",
    "classify_update",
    "update_sample",
]


def sigma_for_timestep(
    scheduler,
    timestep: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the sigma table entry nearest to ``timestep`` (0-dim tensor)."""
    timesteps = scheduler.timesteps.detach().to(device=device or scheduler.timesteps.device, dtype=dtype)
    sigmas = scheduler.sigmas.detach().to(device=device or scheduler.timesteps.device, dtype=dtype)
    index = torch.argmin(torch.abs(timesteps - float(timestep)))
    return sigmas[index]


def predict_clean(
    scheduler,
    prediction: torch.Tensor,
    sample: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """x_0 = x_t - sigma_t * v, computed in float64 like the reference."""
    sigma = sigma_for_timestep(scheduler, timestep, device=sample.device)
    return (sample.to(torch.float64) - sigma * prediction.to(torch.float64)).to(prediction.dtype)


def predict_boundary_state(
    scheduler,
    prediction: torch.Tensor,
    sample: torch.Tensor,
    timestep: float,
    boundary_timestep: float,
) -> torch.Tensor:
    """x_b = x_t - (sigma_t - sigma_b) * v for the high-noise expert's handoff.

    The boundary state is the ODE trajectory evaluated at the boundary
    timestep, not the clean image: skipping re-noising does not turn x_0
    into x_b (review audit section 3).
    """
    sigma_t = sigma_for_timestep(scheduler, timestep, device=sample.device)
    sigma_b = sigma_for_timestep(scheduler, boundary_timestep, device=sample.device)
    return (sample.to(torch.float64) - (sigma_t - sigma_b) * prediction.to(torch.float64)).to(prediction.dtype)


def renoise_low(scheduler, clean: torch.Tensor, noise: torch.Tensor, timestep: float) -> torch.Tensor:
    """x' = (1 - sigma) * clean + sigma * noise at a low-noise timestep."""
    sigma = sigma_for_timestep(scheduler, timestep, device=noise.device)
    return ((1.0 - sigma) * clean.to(torch.float64) + sigma * noise.to(torch.float64)).to(noise.dtype)


def renoise_high(
    scheduler,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: float,
    boundary_timestep: float,
) -> torch.Tensor:
    """x' = alpha * clean + beta * noise with alpha=(1-sigma)/(1-sigma_b),
    beta=sqrt(sigma^2 - (alpha*sigma_b)^2) at a high-noise timestep.

    This preserves the boundary-referenced covariance the high expert was
    distilled for; the plain low-noise formula is wrong above the boundary.
    """
    sigma = sigma_for_timestep(scheduler, timestep, device=noise.device)
    sigma_b = sigma_for_timestep(scheduler, boundary_timestep, device=noise.device)
    alpha = (1.0 - sigma) / (1.0 - sigma_b)
    beta = torch.sqrt(sigma * sigma - (alpha * sigma_b) ** 2)
    return (alpha * clean.to(torch.float64) + beta * noise.to(torch.float64)).to(noise.dtype)


class DmdUpdateKind:
    """Which sample update a step uses, per the reference denoising loop."""

    HIGH_MIDDLE = "high_middle"  # high expert, next step still high: renoise_high
    HIGH_LAST = "high_last"  # high expert's final step: hand the boundary state through unchanged
    LOW_MIDDLE = "low_middle"  # low expert, next step still low: renoise_low
    LOW_LAST = "low_last"  # low expert's final step: emit the clean prediction
    CLEAN = "clean"  # t=0 context pass: publishes KV, never updates the sample


def classify_update(
    *,
    timestep: float,
    next_timestep: float | None,
    boundary_timestep: float | None,
) -> str:
    """Classify a step's update kind from its timestep and the next one.

    ``boundary_timestep=None`` means a single-expert checkpoint: every step is
    low. The handoff step is the last high step when a boundary exists.
    """
    if next_timestep is None:
        return DmdUpdateKind.CLEAN if timestep == 0.0 else DmdUpdateKind.LOW_LAST
    is_high = boundary_timestep is not None and timestep >= boundary_timestep
    next_is_high = boundary_timestep is not None and next_timestep >= boundary_timestep
    if is_high:
        return DmdUpdateKind.HIGH_MIDDLE if next_is_high else DmdUpdateKind.HIGH_LAST
    return DmdUpdateKind.LOW_MIDDLE


def update_sample(
    scheduler,
    *,
    prediction: torch.Tensor,
    sample: torch.Tensor,
    timestep: float,
    next_timestep: float | None,
    noise: torch.Tensor | None,
    boundary_timestep: float | None,
) -> torch.Tensor:
    """The minimal model-side sample-update interface shared by both paths.

    One function, one place, implementing the reference contract:
    high-middle steps renoise through the boundary-aware formula, the last
    high step hands the boundary state to the low expert unchanged, low
    middle steps renoise the clean prediction, and the final low step emits
    it. ``noise`` must be provided exactly when the kind re-noises; the
    clean pass (t=0) never reaches this function (the runner never updates
    the sample from it).
    """
    kind = classify_update(timestep=timestep, next_timestep=next_timestep, boundary_timestep=boundary_timestep)
    if kind == DmdUpdateKind.HIGH_LAST:
        if boundary_timestep is None:
            raise RuntimeError("Boundary handoff requires a boundary timestep")
        return predict_boundary_state(scheduler, prediction, sample, timestep, boundary_timestep)
    if kind == DmdUpdateKind.LOW_LAST:
        return predict_clean(scheduler, prediction, sample, timestep)
    if kind == DmdUpdateKind.HIGH_MIDDLE:
        if noise is None:
            raise RuntimeError(f"{kind} requires a re-noising tensor")
        clean = predict_clean(scheduler, prediction, sample, timestep)
        if boundary_timestep is None:
            raise RuntimeError(f"{kind} requires a boundary timestep")
        return renoise_high(scheduler, clean, noise, next_timestep, boundary_timestep)
    if kind == DmdUpdateKind.LOW_MIDDLE:
        if noise is None:
            raise RuntimeError(f"{kind} requires a re-noising tensor")
        clean = predict_clean(scheduler, prediction, sample, timestep)
        return renoise_low(scheduler, clean, noise, next_timestep)
    raise RuntimeError(f"update_sample does not handle kind {kind!r}")
