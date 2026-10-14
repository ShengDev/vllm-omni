# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan single-stage diffusion topology (Noisy KV).

Diffusion implementation: ``vllm_omni.diffusion.models.causvid_wan``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

CAUSVID_WAN_PIPELINE = PipelineConfig(
    model_type="causvid_wan",
    default_deploy_config_name="causvid_wan.yaml",
    model_arch="CausVidWanPipeline",
    diffusers_class_name="CausVidWanPipeline",
    diffusers_class_aliases=("WanPipeline",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="CausVidWanPipeline",
        ),
    ),
)
