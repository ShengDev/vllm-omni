# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from typing import Any, ClassVar, cast

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.chunk_pipeline_parallel import (
    CHUNK_CONDITIONING_KEY,
    CHUNK_FRAMES_KEY,
    CHUNK_SCHEDULE_KEY,
    KV_HISTORY_CHUNKS_KEY,
    KV_SOURCE_POLICY_KEY,
    run_chunk_pipeline,
)
from vllm_omni.diffusion.distributed.parallel_state import get_pp_group
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin, is_forward_context_available
from vllm_omni.diffusion.lora.loader import WanLoraLoaderMixin
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.causal_dmd import update_sample
from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanSelfAttention, WanTransformer3DModel
from vllm_omni.diffusion.offloader import OffloadPlan
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)
DEBUG_PERF = False
WAN_SAMPLE_SOLVER_CHOICES = {"unipc", "euler"}
FASTWAN_DMD_TIMESTEPS = (1000.0, 757.0, 522.0)
FASTWAN_DMD_SCHEDULER_SHIFT = 8.0
CAUSALWAN_DMD_TIMESTEPS = (1000.0, 850.0, 700.0, 550.0, 350.0, 275.0, 200.0, 125.0)
CAUSALWAN_DMD_SCHEDULER_SHIFT = 12.0
WAN_DMD_CLASS_NAMES = frozenset({"WanDMDPipeline", "WanCausalDMDPipeline"})
CAUSALWAN_DMD_CLASS_NAMES = frozenset({"WanCausalDMDPipeline"})


def build_wan_scheduler(sample_solver: str, flow_shift: float) -> Any:
    if sample_solver == "unipc":
        return FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )
    if sample_solver == "euler":
        return WanEulerScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )

    raise ValueError(
        f"Unsupported Wan sample_solver: {sample_solver}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}"
    )


def load_wan_weights_with_optional_gate(model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load Wan weights and discard optional VSA gates absent from the checkpoint."""
    gate_param_names = {name for name, _ in model.named_parameters() if ".to_gate_compress." in name}
    has_gate_compress_weights = False

    def tracked_weights():
        nonlocal has_gate_compress_weights
        for name, weight in weights:
            if ".to_gate_compress." in name:
                has_gate_compress_weights = True
            yield name, weight

    loaded_weights = AutoWeightsLoader(model).load_weights(tracked_weights())
    setattr(model, "has_gate_compress_weights", has_gate_compress_weights)
    if not has_gate_compress_weights:
        for module in model.modules():
            if isinstance(module, WanSelfAttention):
                module.to_gate_compress = None

    # Optional gate parameters are absent from public FullAttn/DMD checkpoints.
    loaded_weights.update(gate_param_names)
    return loaded_weights


def resolve_wan_sample_solver(req: OmniDiffusionRequest, default: str = "unipc") -> str:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw = extra_args.get("sample_solver", default)
    sample_solver = str(raw).strip().lower()
    if sample_solver not in WAN_SAMPLE_SOLVER_CHOICES:
        raise ValueError(f"Invalid sample_solver={raw!r}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}")
    return sample_solver


def resolve_wan_flow_shift(req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> float:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw_flow_shift = extra_args.get("flow_shift")
    if raw_flow_shift is None:
        raw_flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0

    try:
        return float(raw_flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flow_shift={raw_flow_shift!r}. flow_shift must be a float.") from exc


def resolve_wan_guidance_scales(
    sampling_params: OmniDiffusionSamplingParams,
    default_guidance_scale: float,
) -> tuple[float, float]:
    guidance_scale = (
        sampling_params.guidance_scale if sampling_params.guidance_scale_provided else default_guidance_scale
    )
    guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
    guidance_high = (
        sampling_params.guidance_scale_2
        if sampling_params.guidance_scale_2_provided and sampling_params.guidance_scale_2 is not None
        else (
            guidance_scale[1] if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1 else guidance_low
        )
    )
    return guidance_low, guidance_high


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from vllm_omni.transformers_utils.repo_utils import hf_api

            config_path = hf_api().hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
) -> WanTransformer3DModel:
    """Create WanTransformer3DModel from config dict."""
    kwargs: dict = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
    if "pos_embed_seq_len" in config:
        kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])

    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    if prefix:
        kwargs["prefix"] = prefix

    return WanTransformer3DModel(**kwargs)


def get_wan22_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if sampling_params is not None and sampling_params.output_type is not None:
            output_type = sampling_params.output_type
        if output_type == "latent":
            return video
        video_metadata = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            video_metadata["video_fps_multiplier"] = multiplier
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {"video": video_metadata} if video_metadata else {},
        }

    return post_process_func


def get_wan22_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for Wan2.2: optionally load and resize input image for I2V mode."""
    import numpy as np

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        has_image = raw_image is not None and (not isinstance(raw_image, list) or bool(raw_image))
        request.batch_compatibility_key = ("wan22_image_condition", has_image)
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}

        if raw_image is None:
            request.prompt = prompt
            return request

        if not isinstance(raw_image, (str, PIL.Image.Image)):
            raise TypeError(
                f"""Unsupported image format {raw_image.__class__}.""",
                """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
            )
        image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

        # Calculate dimensions based on aspect ratio if not provided
        if request.sampling_params.height is None or request.sampling_params.width is None:
            # Default max area for 720P
            max_area = 720 * 1280
            aspect_ratio = image.height / image.width

            # Calculate dimensions maintaining aspect ratio
            mod_value = 16  # Must be divisible by 16
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        # Resize image to target dimensions
        image = image.resize(
            (request.sampling_params.width, request.sampling_params.height),  # type: ignore # Above has ensured that width & height are not None
            PIL.Image.Resampling.LANCZOS,
        )
        prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence already checked above

        request.prompt = prompt
        return request

    return pre_process_func


_WAN_TEXT_ENCODER_OFFLOAD_PLAN = OffloadPlan(
    encoder_component_types={"text_encoder": "text_encoder"},
    encoder_block_attrs={"text_encoder": ("encoder.block",)},
    encoder_dlo_weight_replication=frozenset({"text_encoder"}),
)


class Wan22Pipeline(
    nn.Module,
    PipelineParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    DenoiseProgressMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
    WanLoraLoaderMixin,
):
    supports_request_batch = True
    _dit_modules: ClassVar[list[str]] = ["transformer", "transformer_2"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _offload_plan = _WAN_TEXT_ENCODER_OFFLOAD_PLAN

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.chunk_pipeline_mode = getattr(od_config.parallel_config, "pipeline_parallel_mode", "layer") == "chunk"

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Read model_index.json to detect expand_timesteps mode (for TI2V-5B)
        self.expand_timesteps = False
        self.is_dmd = False
        self.is_causalwan_dmd = False
        self.has_transformer_2 = False
        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    class_name = model_index.get("_class_name")
                    self.is_dmd = class_name in WAN_DMD_CLASS_NAMES
                    self.is_causalwan_dmd = class_name in CAUSALWAN_DMD_CLASS_NAMES
            # Check if this is a two-stage model (MoE with transformer_2)
            transformer_2_path = os.path.join(model, "transformer_2")
            self.has_transformer_2 = os.path.exists(transformer_2_path)
        else:
            # For remote models, download and read model_index.json
            try:
                from vllm_omni.transformers_utils.repo_utils import hf_api

                model_index_path = hf_api().hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    class_name = model_index.get("_class_name")
                    self.is_dmd = class_name in WAN_DMD_CLASS_NAMES
                    self.is_causalwan_dmd = class_name in CAUSALWAN_DMD_CLASS_NAMES
                    # Check transformer_2 from model_index
                    transformer_2_info = model_index.get("transformer_2", [None, None])
                    self.has_transformer_2 = transformer_2_info[0] is not None
            except Exception:
                pass

        if self.chunk_pipeline_mode and not self.is_dmd:
            raise ValueError("Chunk pipeline mode currently requires a DMD Wan checkpoint")
        self.boundary_ratio = od_config.boundary_ratio

        # Determine which transformers to load based on boundary_ratio
        # boundary_ratio=1.0: only load transformer_2 (low-noise stage only)
        # boundary_ratio=0.0: only load transformer (high-noise stage only)
        # otherwise: load both transformers
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = self.has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        # Set up weights sources for transformer(s)
        self.weights_sources = []
        if load_transformer:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            )
        if load_transformer_2:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        component_subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(
            model,
            component_subfolders,
            local_files_only=local_files_only,
        )

        # ``from_pretrained_with_prefetch`` re-prefetches and retries if the
        # cache is still half-written (the missing-shard ``OSError`` and the
        # default-``UMT5Config`` size-mismatch ``RuntimeError`` seen on multi
        # -worker HSDP / ring launches), instead of crashing the worker.
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        # Initialize transformers with correct config (weights loaded via load_weights)
        if load_transformer:
            transformer_config = load_transformer_config(model, "transformer", local_files_only)
            self.transformer = self._create_transformer(transformer_config)
        else:
            self.transformer = None

        if load_transformer_2:
            transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
            self.transformer_2 = self._create_transformer(transformer_2_config)
        else:
            self.transformer_2 = None

        for transformer in (self.transformer, self.transformer_2):
            if transformer is not None:
                transformer.preserve_vsa_all_blocks = self.is_dmd

        # Store the active transformer config
        if load_transformer:
            self.transformer_config = self.transformer.config
        elif load_transformer_2:
            self.transformer_config = self.transformer_2.config
        else:
            raise RuntimeError("No transformer loaded")

        self._sample_solver = "euler" if self.is_dmd else "unipc"
        if self.is_causalwan_dmd:
            default_dmd_shift = CAUSALWAN_DMD_SCHEDULER_SHIFT
        elif self.is_dmd:
            default_dmd_shift = FASTWAN_DMD_SCHEDULER_SHIFT
        else:
            default_dmd_shift = None
        self._flow_shift = (
            default_dmd_shift
            if default_dmd_shift is not None
            else od_config.flow_shift
            if od_config.flow_shift is not None
            else 5.0
        )
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        """Create a transformer from a config dict. Respects od_config.quantization_config."""
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(config, quant_config=quant_config)

    def _dmd_timesteps(self) -> tuple[float, ...]:
        if self.is_causalwan_dmd:
            return CAUSALWAN_DMD_TIMESTEPS
        return FASTWAN_DMD_TIMESTEPS

    def _active_layer_range(self) -> tuple[int, int]:
        # Both towers shard identically (same layer count and config), so the
        # chunk pipeline ranges are interchangeable; prefer the loaded one.
        transformer = self.transformer if self.transformer is not None else self.transformer_2
        if transformer is None:
            raise RuntimeError("No transformer available")
        return (transformer.start_layer, transformer.end_layer)

    def _select_dit(self, timestep) -> WanTransformer3DModel:
        """Single-timestep tower lookup; a test oracle for ``_dit_router``.

        Production paths never call this: the chunk pipeline uses the
        precomputed ``_dit_router`` table and the native path resolves the
        tower inside ``diffuse()``. This method exists so tests can pin the
        routing semantics (boundary comparison, single-tower fallback)
        against the table on plain values.
        """
        value = timestep.reshape(-1)[0] if isinstance(timestep, torch.Tensor) else timestep
        t = float(value)
        boundary = self._request_boundary_timestep()
        if t < boundary and self.transformer_2 is not None:
            return self.transformer_2
        if self.transformer is not None:
            return self.transformer
        if self.transformer_2 is not None:
            return self.transformer_2
        raise RuntimeError("No transformer available")

    def _request_boundary_timestep(self) -> float:
        """Engine-configured boundary wins; fall back to the Wan2.2 default."""
        if self.boundary_ratio is not None:
            return self.boundary_ratio * self.scheduler.config.num_train_timesteps
        return 0.875 * self.scheduler.config.num_train_timesteps

    def _dit_router(self, timesteps: torch.Tensor, boundary_timestep: float):
        """Precompute the per-step tower choice from materialized timestep values.

        Index i selects the tower for denoise step i; the extra clean pass
        (t=0, index len(timesteps)) always routes to the low-noise tower.
        No device-to-host sync per slot: the values are materialized once.
        """
        values = tuple(float(value) for value in timesteps.detach().cpu().flatten().tolist())
        if self.transformer_2 is None:
            if self.transformer is None:
                raise RuntimeError("No transformer available")
            return [self.transformer] * (len(values) + 1)
        if self.transformer is None:
            return [self.transformer_2] * (len(values) + 1)
        return [self.transformer_2 if value < boundary_timestep else self.transformer for value in values] + [
            self.transformer_2
        ]

    def _resolve_temporal_chunks(self, *, latent_frames: int, extra_args: dict | None) -> tuple[int, int, int]:
        """Return (chunks, chunk_latent_frames, chunk_pixel_frames).

        Omitting ``chunk_frames`` treats the whole clip as one chunk (full-video path).
        """
        scale = self.vae_scale_factor_temporal
        default_frames = (latent_frames - 1) * scale + 1
        chunk_frames = int((extra_args or {}).get(CHUNK_FRAMES_KEY, default_frames))
        if chunk_frames < 1 or (chunk_frames - 1) % scale:
            raise ValueError("chunk_frames must be positive and 1 modulo the VAE temporal scale")
        chunk_t = min((chunk_frames - 1) // scale + 1, latent_frames)
        if latent_frames % chunk_t:
            raise ValueError("Total output latent frames must be divisible by the chunk latent length")
        return latent_frames // chunk_t, chunk_t, chunk_frames

    def _diffuse_chunks(self, req, latents, timesteps, prompt_embeds, dtype, generator, boundary_timestep=None):
        if not self.is_dmd:
            raise ValueError("Chunk pipeline currently requires a DMD Wan checkpoint")
        parallel = self.od_config.parallel_config
        if parallel.pipeline_parallel_size < 1:
            raise ValueError("Chunk pipeline requires pipeline_parallel_size >= 1")
        if getattr(self.od_config, "step_execution", False) or getattr(self.od_config, "streaming_output", False):
            raise ValueError("Chunk pipeline requires step_execution=False and streaming_output=False")
        incompatible = [
            name
            for name in (
                "tensor_parallel_size",
                "sequence_parallel_size",
                "cfg_parallel_size",
                "vae_patch_parallel_size",
                "text_encoder_tp_size",
            )
            if (getattr(parallel, name, 1) or 1) != 1
        ]
        if parallel.data_parallel_size not in (None, 1):
            incompatible.append("data_parallel_size")
        if parallel.use_hsdp:
            incompatible.append("use_hsdp")
        if parallel.enable_expert_parallel:
            incompatible.append("enable_expert_parallel")
        if incompatible:
            raise ValueError("Chunk pipeline unsupported with: " + ", ".join(incompatible))
        params = req.sampling_params_list[0]
        if req.num_reqs != 1 or params.num_outputs_per_prompt != 1:
            raise ValueError("Chunk pipeline mode currently accepts one video request at a time")
        extra = params.extra_args or {}
        chunks, chunk_t, _chunk_frames = self._resolve_temporal_chunks(
            latent_frames=latents.shape[2], extra_args=extra
        )
        if chunks < 2:
            raise ValueError("Chunk pipeline requires more than one temporal chunk")
        schedule = extra.get(CHUNK_SCHEDULE_KEY, "stepwise")
        if isinstance(generator, list):
            generator = generator[0]
        seed = params.seed if params.seed is not None else generator.initial_seed() if generator is not None else 0
        shape = (*latents.shape[:2], chunk_t, *latents.shape[3:])
        use_kv = extra.get(CHUNK_CONDITIONING_KEY) == "latest_kv"
        step_noises = None
        if use_kv:
            step_noises = [
                randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=dtype)
                for _ in range(len(timesteps) - 1)
            ]
        # ``record_denoise_step`` normally turns a CUDA scalar into a Python
        # float. In chunk PP that happens once per local slot and forces the
        # host to synchronize the current stream. The scheduler timesteps are
        # immutable for this request, so materialize their normalized values
        # once before the chunk pipeline starts instead.
        num_train_timesteps = getattr(getattr(self.scheduler, "config", None), "num_train_timesteps", None)
        normalized_timesteps = None
        if num_train_timesteps and is_forward_context_available():
            normalized_timesteps = tuple(
                float(value) / float(num_train_timesteps) for value in timesteps.detach().cpu().flatten().tolist()
            )
        # Resolve the effective boundary once per request (engine config wins,
        # mirroring the native path) and precompute the per-step tower choice
        # so the slot loop never reads a CUDA scalar.
        if boundary_timestep is None:
            boundary_timestep = self._request_boundary_timestep()
        tower_per_step = self._dit_router(timesteps, boundary_timestep)

        def predict_noise(model_input, timestep, temporal_offset, step_idx, intermediate_tensors=None, kv_context=None):
            self._current_timestep = timestep[0]
            if normalized_timesteps is None:
                self.record_denoise_step(step_idx, timestep[0])
            else:
                # The KV clean pass uses a synthetic t=0 at the extra final
                # step index, past the normalized denoise timesteps.
                normalized = normalized_timesteps[step_idx] if step_idx < len(normalized_timesteps) else 0.0
                self.record_denoise_step(step_idx, normalized_timestep=normalized)
            return self.predict_noise(
                current_model=tower_per_step[step_idx],
                hidden_states=model_input.to(dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                temporal_offset=temporal_offset,
                intermediate_tensors=intermediate_tensors,
                kv_context=kv_context,
                return_dict=False,
            )

        def update_sample_for_chunk(
            prediction, sample, timestep, next_timestep, noise, boundary_timestep=boundary_timestep
        ):
            return update_sample(
                self.scheduler,
                prediction=prediction,
                sample=sample,
                timestep=timestep,
                next_timestep=next_timestep,
                noise=noise,
                boundary_timestep=boundary_timestep,
            )

        result, self.chunk_pipeline_metrics = run_chunk_pipeline(
            predict_noise=predict_noise,
            update_sample=update_sample_for_chunk,
            timesteps=timesteps,
            shape=shape,
            chunks=chunks,
            seed=seed,
            device=latents.device,
            schedule=schedule,
            layer_range=self._active_layer_range(),
            initial_latents=latents if use_kv else None,
            step_noises=step_noises,
            kv_history_chunks=int(extra.get(KV_HISTORY_CHUNKS_KEY, 6)) if use_kv else None,
            kv_source_policy=extra.get(KV_SOURCE_POLICY_KEY, "latest"),
        )
        return result

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any],
        latent_condition: torch.Tensor | None = None,
        first_frame_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        profile = getattr(self, "_collect_pp_metrics", False)
        # ``record_denoise_step`` materializes a CUDA scalar only when a
        # request ForwardContext is active. Resolve immutable scheduler values
        # once in that case (or when metrics need their host-side copy), rather
        # than synchronizing once per native PP stage forward.
        normalized_timesteps = None
        timestep_values = None
        if profile or is_forward_context_available():
            timestep_values = tuple(float(value) for value in timesteps.detach().cpu().flatten().tolist())
            ntt = getattr(getattr(self.scheduler, "config", None), "num_train_timesteps", None)
            if ntt:
                normalized_timesteps = tuple(value / float(ntt) for value in timestep_values)
        if profile:
            pp = get_pp_group()
            current_omni_platform.synchronize()
            if pp.world_size > 1:
                torch.distributed.barrier(group=pp.device_group)
            torch.accelerator.reset_peak_memory_stats(latents.device)
            self._full_pp_records = []
            # Resolve CUDA events only after the request. Resolving them per
            # stage forward changes the native PP baseline being measured.
            self._full_pp_pending_timings = []
            self._full_pp_wait_ms = 0.0
            self._full_pp_feedback_bytes = 0
            self._full_pp_origin = torch.cuda.Event(enable_timing=True)
            self._full_pp_origin.record()
            assert timestep_values is not None
            self._full_pp_timestep_values = timestep_values
            profile_start = time.perf_counter()
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                if profile:
                    wait_start = time.perf_counter()
                    if isinstance(latents, AsyncLatents):
                        latents = latents._resolve()
                    self._sync_pp_send()
                    self._full_pp_wait_ms += (time.perf_counter() - wait_start) * 1000
                    self._full_pp_step_idx = step_idx
                self._current_timestep = t
                if normalized_timesteps is None:
                    self.record_denoise_step(step_idx, t)
                else:
                    self.record_denoise_step(step_idx, normalized_timestep=normalized_timesteps[step_idx])

                # Select model based on timestep and boundary_ratio
                # High noise stage (t >= boundary_timestep): use transformer
                # Low noise stage (t < boundary_timestep): use transformer_2
                if boundary_timestep is not None and t < boundary_timestep:
                    # Low noise stage - always use guidance_high for this stage
                    current_guidance_scale = guidance_high
                    if self.transformer_2 is not None:
                        current_model = self.transformer_2
                    elif self.transformer is not None:
                        # Fallback to transformer if transformer_2 not loaded
                        current_model = self.transformer
                    else:
                        raise RuntimeError("No transformer available for low-noise stage")
                else:
                    # High noise stage - always use guidance_low for this stage
                    current_guidance_scale = guidance_low
                    if self.transformer is not None:
                        current_model = self.transformer
                    elif self.transformer_2 is not None:
                        # Fallback to transformer_2 if transformer not loaded
                        current_model = self.transformer_2
                    else:
                        raise RuntimeError("No transformer available for high-noise stage")

                if self.expand_timesteps and latent_condition is not None:
                    # I2V mode: blend condition with latents using mask
                    latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(dtype)

                    # Expand timesteps per patch - use floor division to match patch embedding
                    patch_size = self.transformer_config.patch_size
                    patch_height = latents.shape[3] // patch_size[1]
                    patch_width = latents.shape[4] // patch_size[2]

                    # Create mask at patch resolution (same as hidden states sequence length)
                    patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                    patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]  # Ensure correct dimensions
                    temp_ts = (patch_mask[0][0] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # T2V mode: standard forward
                    latent_model_input = latents.to(dtype)
                    timestep = t.expand(latents.shape[0])

                do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": current_model,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                if self.is_dmd:

                    def update_dmd(prediction, sample):
                        next_t = float(timesteps[step_idx + 1]) if step_idx + 1 < len(timesteps) else None
                        noise = (
                            randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                            if next_t is not None
                            else None
                        )
                        return update_sample(
                            self.scheduler,
                            prediction=prediction,
                            sample=sample,
                            timestep=float(t),
                            next_timestep=next_t,
                            noise=noise,
                            boundary_timestep=self._request_boundary_timestep(),
                        )

                    latents = self.dmd_step_maybe_with_pp(noise_pred, latents, update_dmd)
                else:
                    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                if profile and pp.world_size > 1 and pp.is_last_rank:
                    self._full_pp_feedback_bytes += latents.numel() * latents.element_size()
                pbar.update()

        if profile:
            wait_start = time.perf_counter()
            if isinstance(latents, AsyncLatents):
                latents = latents._resolve()
            self._sync_pp_send()
            self._full_pp_wait_ms += (time.perf_counter() - wait_start) * 1000
            current_omni_platform.synchronize()
            if pp.world_size > 1:
                torch.distributed.barrier(group=pp.device_group)
            for record, start, end in self._full_pp_pending_timings:
                record["forward_ms"] = start.elapsed_time(end)
                record["stage_start_ms"] = self._full_pp_origin.elapsed_time(start)
                record["stage_end_ms"] = self._full_pp_origin.elapsed_time(end)
            wall_ms = (time.perf_counter() - profile_start) * 1000
            payload = {
                "rank": pp.rank_in_group,
                "stage_idx": pp.rank_in_group,
                "layer_range": [self.transformer.start_layer, self.transformer.end_layer],
                "steps": self._full_pp_records,
                "slots": [],
                "denoise_wall_ms": wall_ms,
                "comm_wait_ms": self._full_pp_wait_ms,
                "comm_wait_scope": "explicit tensor receive and pending-send waits; metadata/dispatch remain in wall",
                "denoise_peak_allocated_bytes": torch.accelerator.max_memory_allocated(latents.device),
                "activation_bytes_sent": sum(r["activation_bytes_sent"] for r in self._full_pp_records),
                "sample_bytes_sent": 0,
                "feedback_bytes_sent": self._full_pp_feedback_bytes,
                "condition_bytes_sent": 0,
            }
            ranks = [payload]
            if pp.world_size > 1:
                ranks = [None] * pp.world_size
                torch.distributed.all_gather_object(ranks, payload, group=pp.cpu_group)
            self.chunk_pipeline_metrics = {
                "execution": "full_video_layer_pipeline",
                "schedule": "full",
                "gap": 0,
                "lag": 0,
                "world_size": pp.world_size,
                "chunks": 1,
                "chunk_latent_frames": latents.shape[2],
                "cond_frames": 0,
                "latent_shape": list(latents.shape),
                "ranks": ranks,
                "denoise_wall_ms": max(r["denoise_wall_ms"] for r in ranks),
                "forward_total_ms": sum(s["forward_ms"] for r in ranks for s in r["steps"]),
                "comm_wait_rank_sum_ms": sum(r["comm_wait_ms"] for r in ranks),
                "condition_bytes_sent": 0,
                "latent_gather_ms": 0.0,
            }
            self._full_pp_records = None
            self._full_pp_pending_timings = None
            self._full_pp_timestep_values = None
        return latents

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        sampling_params_list = req.sampling_params_list
        common = sampling_params_list[0]
        self._collect_pp_metrics = bool((common.extra_args or {}).get("collect_pp_metrics", False))
        self._full_pp_records = None
        prompt_texts = [prompt if isinstance(prompt, str) else (prompt.get("prompt") or "") for prompt in req.prompts]
        negative_prompts = [
            None if isinstance(prompt, str) else prompt.get("negative_prompt") for prompt in req.prompts
        ]
        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            req.prompts,
            {
                "prompt_embeds": None,
                "negative_prompt_embeds": None,
            },
        )
        prompt_embeds = prompt_fields["prompt_embeds"]
        negative_prompt_embeds = prompt_fields["negative_prompt_embeds"]
        prompt: list[str] | None = prompt_texts if prompt_embeds is None else None
        negative_prompt: list[str] | None = None
        if negative_prompt_embeds is None and any(value is not None for value in negative_prompts):
            negative_prompt = [value or "" for value in negative_prompts]

        if prompt is not None and not all(prompt):
            raise ValueError("Prompt is required for Wan2.2 generation when prompt_embeds are not provided.")

        height = common.height or 480
        width = common.width or 832
        num_frames = common.num_frames or 81

        # Ensure dimensions are compatible with VAE and patch size
        # For expand_timesteps mode, we need latent dims to be even (divisible by patch_size)
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        if self.is_dmd:
            # The checkpoint was distilled for these transitions. Ignore
            # request-level step counts, including the engine's 1-step warmup.
            num_steps = len(self._dmd_timesteps())
        else:
            num_steps = 40 if common.num_inference_steps is None else common.num_inference_steps

        output_type = common.output_type or "np"
        num_outputs_per_prompt = common.num_outputs_per_prompt or 1
        attention_kwargs: dict | None = None

        guidance_low, guidance_high = resolve_wan_guidance_scales(
            common,
            default_guidance_scale=1.0 if self.is_dmd else 4.0,
        )
        if self.chunk_pipeline_mode and (guidance_low != 1.0 or guidance_high != 1.0):
            raise ValueError("Chunk pipeline mode currently requires guidance_scale=1")

        # record guidance for properties
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Prefer engine-configured boundary_ratio, but allow per-request fallback.
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else common.boundary_ratio

        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for T2V generation. using default value 0.875")

        # validate shapes
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high if boundary_ratio is not None else None,
            boundary_ratio=boundary_ratio,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        # Get dtype from whichever transformer is loaded
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            # Fallback to text_encoder dtype if no transformer loaded
            dtype = self.text_encoder.dtype

        generator = req.collate_request_generators(num_outputs_per_prompt, None)
        request_latents = req.collate_request_tensors("latents", None)
        if self.chunk_pipeline_mode and request_latents is not None:
            raise ValueError("Chunk pipeline mode currently prepares its own per-chunk seeded noise")

        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            # Sync GPU before timing to ensure accurate measurements
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start
        do_classifier_free_guidance = guidance_low > 1.0 or guidance_high > 1.0
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_outputs_per_prompt,
                max_sequence_length=common.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            prompt_embeds = prompt_embeds.repeat_interleave(num_outputs_per_prompt, dim=0)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_outputs_per_prompt, dim=0)
            elif do_classifier_free_guidance:
                _, negative_prompt_embeds = self.encode_prompt(
                    prompt=[""] * req.num_reqs,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=num_outputs_per_prompt,
                    max_sequence_length=common.max_sequence_length or 512,
                    device=device,
                    dtype=dtype,
                )

        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        if self.is_dmd:
            timesteps = torch.tensor(self._dmd_timesteps(), device=device, dtype=torch.float32)
        else:
            first_request = req.requests[0]
            sample_solver = resolve_wan_sample_solver(first_request, default=self._sample_solver)
            flow_shift = resolve_wan_flow_shift(first_request, self.od_config)
            if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
                self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
                self._sample_solver = sample_solver
                self._flow_shift = flow_shift

            self.scheduler.set_timesteps(num_steps, device=device)
            timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            _t_latent_prep_start = time.perf_counter()
        images: list[PIL.Image.Image | torch.Tensor | None] = []
        for request_prompt in req.prompts:
            multi_modal_data = request_prompt.get("multi_modal_data", {}) if not isinstance(request_prompt, str) else {}
            raw_image = multi_modal_data.get("image")
            if isinstance(raw_image, list):
                if len(raw_image) > 1:
                    logger.warning("Received multiple images for one Wan request; using only the first image.")
                raw_image = raw_image[0] if raw_image else None
            if isinstance(raw_image, str):
                raw_image = PIL.Image.open(raw_image)
            images.append(cast(PIL.Image.Image | torch.Tensor | None, raw_image))

        latent_condition = None
        first_frame_mask = None
        if self.chunk_pipeline_mode and any(image is not None for image in images):
            raise ValueError("Chunk pipeline mode currently supports text-to-video requests")

        if self.expand_timesteps and any(image is not None for image in images):
            if not all(image is not None for image in images):
                raise ValueError("Cannot batch Wan requests with a mix of provided and missing image conditions.")
            # I2V mode: encode image and prepare condition
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            image_tensors = []
            for image in images:
                assert image is not None
                if isinstance(image, PIL.Image.Image):
                    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                    image_tensor = video_processor.preprocess(image, height=height, width=width)
                else:
                    image_tensor = image.unsqueeze(0) if image.ndim == 3 else image
                image_tensors.append(image_tensor)
            image_tensor = DiffusionRequestBatch.collate_tensors(image_tensors, "image condition", None)
            assert image_tensor is not None
            image_tensor = image_tensor.repeat_interleave(num_outputs_per_prompt, dim=0)

            # Use out_channels for noise latents (not in_channels which includes condition)
            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]

            # Prepare noise latents
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=request_latents,
            )

            # Encode image condition
            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]

            image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")

            # Normalize condition latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std
            latent_condition = latent_condition.to(torch.float32)

            # Create mask: 0 for first frame (condition), 1 for rest (to denoise)
            first_frame_mask = torch.ones(
                batch_size, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode: standard latent preparation
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=request_latents,
            )
        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if attention_kwargs is None:
            attention_kwargs = {}

        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            _t_denoise_start = time.perf_counter()
        if self.chunk_pipeline_mode:
            latents = self._diffuse_chunks(
                req, latents, timesteps, prompt_embeds, dtype, generator, boundary_timestep=boundary_timestep
            )
        else:
            latents = self.diffuse(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                boundary_timestep=boundary_timestep,
                dtype=dtype,
                attention_kwargs=attention_kwargs,
                latent_condition=latent_condition,
                first_frame_mask=first_frame_mask,
                generator=generator,
            )

        # Wan2.2 is prone to out of memory errors when predicting large videos
        # so we empty the cache here to avoid OOM before VAE decoding. A latent
        # response does not decode through the VAE, so avoid an unnecessary
        # allocator flush on that path.
        if output_type != "latent" and current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        # For I2V mode: blend final latents with condition
        if self.expand_timesteps and latent_condition is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            _t_decode_start = time.perf_counter()
        media = None
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            decoded = self.vae.decode(latents, return_dict=False)[0]
            # Distributed VAE decode uses broadcast_result=False, so only the
            # output-owning rank receives the full [B, C, T, H, W] video; other
            # ranks get an empty placeholder. Emit typed media only from the
            # owning rank and keep the placeholder on the legacy output field, so
            # the media batch-dimension check in split_diffusion_output_by_request
            # does not trip on every non-owner rank.
            if decoded is not None and decoded.dim() == 5:
                output = None
                media = DiffusionMediaOutput(
                    video=VideoMediaOutput(
                        tensor=decoded,
                        spec=VideoTensorSpec(
                            layout=VideoTensorLayout.BCTHW,
                            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
                        ),
                    )
                )
            else:
                output = decoded

        if DEBUG_PERF or self.chunk_pipeline_mode or self._collect_pp_metrics:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms

            if _is_rank_zero():
                logger.info(
                    "Pipeline stage timing summary: "
                    "TextEncoding=%.2f ms, LatentPreparation=%.2f ms, "
                    "Denoising=%.2f ms (%d steps), Decoding=%.2f ms, "
                    "StagesSum=%.2f ms, PipelineWall=%.2f ms, Unaccounted=%.2f ms",
                    _t_text_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                    _t_pipeline_wall_ms - _t_stages_sum,
                )

        if (self.chunk_pipeline_mode or self._collect_pp_metrics) and _is_rank_zero():
            self.chunk_pipeline_metrics.update(
                {
                    "seed": common.seed,
                    "experiment_iteration": (common.extra_args or {}).get("experiment_iteration"),
                    "model_weight_layout": "layer_partition",
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "output_type": output_type,
                    "torch": torch.__version__,
                    "dtype": str(dtype),
                    "gpu": torch.cuda.get_device_name(self.device),
                    "transformer_layer_range": [self.transformer.start_layer, self.transformer.end_layer],
                    "stage_ms": {
                        "text_encode": _t_text_enc_ms,
                        "latent_prep": _t_latent_prep_ms,
                        "denoise_and_gather": _t_denoise_ms,
                        "vae_decode": _t_decode_ms,
                        "pipeline_wall": _t_pipeline_wall_ms,
                    },
                }
            )
            logger.info("CHUNK_PP_METRICS %s", json.dumps(self.chunk_pipeline_metrics))

        return split_diffusion_output_by_request(
            DiffusionOutput(
                output=output,
                media=media,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            ),
            req,
            num_outputs_per_prompt=num_outputs_per_prompt,
        )

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        if current_model is None:
            current_model = self.transformer
        records = getattr(self, "_full_pp_records", None)
        if records is not None:
            wait_start = time.perf_counter()
            intermediate = kwargs.get("intermediate_tensors")
            if intermediate is not None:
                # Resolve the incoming activation before starting the GPU
                # compute event, so peer waiting is not called DiT compute.
                intermediate["hidden_states"]
            wait_ms = (time.perf_counter() - wait_start) * 1000
            self._full_pp_wait_ms += wait_ms
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
        result = current_model(**kwargs)
        if records is not None:
            end.record()
            record = {
                "chunk_idx": 0,
                "step_idx": self._full_pp_step_idx,
                "slot_idx": self._full_pp_step_idx,
                "rank": get_pp_group().rank_in_group,
                "stage_idx": get_pp_group().rank_in_group,
                "timestep": self._full_pp_timestep_values[self._full_pp_step_idx],
                "input_latent_frames": kwargs["hidden_states"].shape[2],
                "forward_ms": 0.0,
                "stage_start_ms": 0.0,
                "stage_end_ms": 0.0,
                "time_origin": "rank_local_cuda_event",
                "comm_wait_ms": wait_ms,
                "activation_bytes_sent": sum(t.numel() * t.element_size() for t in result.tensors.values())
                if isinstance(result, IntermediateTensors)
                else 0,
            }
            records.append(record)
            self._full_pp_pending_timings.append((record, start, end))
        return result if isinstance(result, IntermediateTensors) else result[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_wan_weights_with_optional_gate(self, weights)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
        boundary_ratio=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")


# ---------------------------------------------------------------------------
# DMD2-distilled variant
# ---------------------------------------------------------------------------


class WanT2VDMD2Pipeline(DMD2PipelineMixin, Wan22Pipeline):
    """Wan 2.x T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
