# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DreamZero opt-in Noisy PP adapter (Serial / Latest cell ticks).

Default DreamZero stays on the Legacy Session path. When
``enable_noisy_pp=True``, the pipeline binds this adapter and runs
``run_chunk_pipeline`` with NoisyKV instead of paged-session KV.

MVP scope: video DiT cells + UniPC step advance. Action registers / CFG are
disabled on this path (zeros / positive-only); full parity is follow-up work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.models.dreamzero.causal_wan_model import (
    CausalWanModel,
    sinusoidal_embedding_1d,
)
from vllm_omni.experimental.ar_diffusion.chunk_executor import ChunkAdapter

logger = init_logger(__name__)


class DreamZeroNoisyChunkAdapter(ChunkAdapter):
    """Per-cell ``embed → block → unembed`` over :class:`CausalWanModel` + NoisyKV."""

    def __init__(
        self,
        transformer: CausalWanModel,
        *,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        video_scheduler: Any,
        current_start_frame: int = 0,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        on_finished: Callable[[int, torch.Tensor], None] | None = None,
    ) -> None:
        self.transformer = transformer
        self.prompt_embeds = prompt_embeds
        self.latent_shape = latent_shape
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.video_scheduler = video_scheduler
        self.current_start_frame = int(current_start_frame)
        self.y = y
        self.clip_feature = clip_feature
        self.on_finished = on_finished
        self.num_denoise_steps = int(timesteps.numel())
        self._timesteps = [timesteps[i].detach() for i in range(self.num_denoise_steps)]
        self.finished: dict[str, dict[int, torch.Tensor]] = {}
        self._live: dict[str, dict[int, torch.Tensor]] = {}
        self._stash: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self._cond: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._last_meta: tuple[str, int, int] | None = None
        self.num_layers = int(transformer.num_layers)
        self.dim = int(transformer.dim)
        self.patch_size = tuple(transformer.patch_size)
        self._seq_len = self.seq_len_for_latent(latent_shape)

    def seq_len_for_latent(self, latent_shape: tuple[int, ...]) -> int:
        if len(latent_shape) != 5:
            raise ValueError(f"expected B,C,T,H,W; got {latent_shape}")
        _, _, t, h, w = latent_shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError(f"latent {latent_shape} not aligned to patch {self.patch_size}")
        return (t // pt) * (h // ph) * (w // pw)

    def _init_noise(self, chunk: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed * 1_000_003 + chunk * 4096)
        return torch.randn(self.latent_shape, generator=gen, device=self.device, dtype=self.dtype)

    @staticmethod
    def _slice_batch(tensor: torch.Tensor, index: int, n_tasks: int) -> torch.Tensor:
        return tensor[index : index + 1] if tensor.shape[0] == n_tasks else tensor

    def stash_activation(
        self,
        payload: dict,
        *,
        handles: list | None = None,
        meta: tuple[str, int, int] | None = None,
    ) -> None:
        del handles
        tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor) and k not in ("chunk", "step")}
        key = meta or self._last_meta
        if key is None:
            raise RuntimeError("stash_activation needs meta")
        self._stash[key] = tensors

    def take_stashed_activation(self, req: str, chunk: int, step: int):
        return self._stash.pop((req, chunk, step), None) or self._stash.pop(("", chunk, step), None)

    def _resolve_latent_hidden(
        self, *, req: str, chunk: int, step: int, index: int, n_tasks: int, hidden: Any, live: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent: torch.Tensor | None = None
        tokens: torch.Tensor | None = None
        src = hidden if isinstance(hidden, dict) else None
        if src is None or "hidden_states" not in src:
            src = self.take_stashed_activation(req, chunk, step)
        if isinstance(src, dict):
            if "latent" in src:
                latent = self._slice_batch(src["latent"], index, n_tasks)
            if "hidden_states" in src:
                tokens = self._slice_batch(src["hidden_states"], index, n_tasks)
        elif isinstance(hidden, torch.Tensor):
            latent = self._slice_batch(hidden, index, n_tasks)
        if latent is None:
            latent = live.get(chunk) or self._init_noise(chunk)
        return latent, tokens

    def _prepare_condition(
        self, *, req: str, chunk: int, step: int, latent: torch.Tensor
    ) -> dict[str, Any]:
        key = (req, chunk, step)
        hit = self._cond.get(key)
        if hit is not None:
            return hit
        tf = self.transformer
        x = latent
        if self.y is not None:
            # Match CausalWan i2v: concat control channels before patch embed.
            x = torch.cat([x, self.y.to(dtype=x.dtype)], dim=1)
        x = tf.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long, device=x.device)
        tokens = x.flatten(start_dim=2).transpose(1, 2)
        seq_len = int(tokens.shape[1])
        if seq_len != self._seq_len:
            raise RuntimeError(f"DreamZero noisy PP seq_len mismatch: got {seq_len}, expected {self._seq_len}")
        freqs = tf._create_freqs(grid_size, self.current_start_frame)
        B = tokens.shape[0]
        nfpb = max(1, int(tf.num_frame_per_block))
        t_scalar = self._timesteps[step].to(device=self.device)
        timestep = torch.ones([B, nfpb], device=self.device, dtype=torch.float32) * t_scalar.float()
        F_t = nfpb
        flat = timestep.unsqueeze(-1).expand(B, F_t, seq_len // F_t).reshape(B, -1)
        e = tf.time_embedding(sinusoidal_embedding_1d(tf.freq_dim, flat.flatten()).type_as(tokens))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = tf.time_projection(e).unflatten(dim=2, sizes=(6, tf.dim))
        context = tf.text_embedding(self.prompt_embeds.to(dtype=tokens.dtype))
        if self.clip_feature is not None and hasattr(tf, "img_emb"):
            context = torch.cat([tf.img_emb(self.clip_feature.to(dtype=tokens.dtype)), context], dim=1)
        cond = {
            "tokens": tokens,
            "e0": e0,
            "e": e,
            "freqs": freqs,
            "grid_size": grid_size,
            "seq_len": seq_len,
            "context": context,
        }
        self._cond[key] = cond
        return cond

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        outs: list[dict[str, torch.Tensor]] = []
        tf = self.transformer
        for i, (req, cell) in enumerate(tasks):
            chunk, step, block = cell
            self._last_meta = (req, chunk, step)
            ctx_list = kv_contexts[i] if i < len(kv_contexts) else None
            kv_ctx = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            live = self._live.setdefault(req, {})
            latent, tokens = self._resolve_latent_hidden(
                req=req, chunk=chunk, step=step, index=i, n_tasks=len(tasks), hidden=hidden, live=live
            )
            if step >= self.num_denoise_steps:
                # Clean pass: keep finished latent; no DiT.
                live[chunk] = latent
                self.finished.setdefault(req, {})[chunk] = latent.detach()
                outs.append({"latent": latent})
                continue
            if block < 0 or block >= self.num_layers:
                raise RuntimeError(f"DreamZero noisy PP block {block} out of range [0, {self.num_layers})")
            if tokens is None or block == 0:
                cond = self._prepare_condition(req=req, chunk=chunk, step=step, latent=latent)
                tokens = cond["tokens"]
            else:
                cond = self._cond.get((req, chunk, step))
                if cond is None:
                    raise RuntimeError(f"cell {(chunk, step, block)} missing condition cache")
            kv_inputs = kv_ctx.to_layer_inputs() if kv_ctx is not None and hasattr(kv_ctx, "to_layer_inputs") else kv_ctx
            tokens, _updated = tf.blocks[block](
                x=tokens,
                e=cond["e0"],
                freqs=cond["freqs"],
                freqs_action=tf.freqs_action,
                freqs_state=tf.freqs_state,
                context=cond["context"],
                action_register_length=None,
                kv_cache=kv_inputs,
                crossattn_cache=None,
                current_start_frame=self.current_start_frame,
            )
            cond["tokens"] = tokens
            if block < self.num_layers - 1:
                live[chunk] = latent
                outs.append({"latent": latent, "hidden_states": tokens})
                continue
            # Last layer: head + unpatchify + scheduler step.
            seq_len = int(cond["seq_len"])
            x_video = tokens[:, :seq_len]
            e_video = cond["e"][:, :seq_len]
            x_video = tf.head(x_video, e_video.unsqueeze(2))
            pred = tf.unpatchify(x_video, cond["grid_size"])
            # UniPC scheduler follows DreamZero legacy layout (B,T,C,H,W).
            t = self._timesteps[step]
            latent_bt = latent.transpose(1, 2)
            pred_bt = pred.transpose(1, 2)
            latent_bt = self.video_scheduler.step(
                pred_bt,
                t,
                latent_bt,
                return_dict=False,
                generator=None,
            )[0]
            latent = latent_bt.transpose(1, 2).to(dtype=self.dtype)
            self._cond.pop((req, chunk, step), None)
            if step == self.num_denoise_steps - 1:
                self.finished.setdefault(req, {})[chunk] = latent.detach()
                if self.on_finished is not None:
                    self.on_finished(chunk, latent)
            live[chunk] = latent
            outs.append({"latent": latent})
        if len(outs) == 1:
            return outs[0]
        keys = outs[0].keys()
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}

    def pack_activation(self, output):
        if isinstance(output, dict):
            return {k: v.contiguous() for k, v in output.items() if isinstance(v, torch.Tensor)}
        if isinstance(output, IntermediateTensors):
            return {k: v.contiguous() for k, v in output.tensors.items()}
        if isinstance(output, torch.Tensor):
            return {"latent": output.contiguous()}
        raise TypeError(f"pack expects dict/tensor, got {type(output)}")

    def activation_spec(self, *, batch: int, include_hidden: bool):
        batch = max(1, int(batch))
        _b, c, t, h, w = self.latent_shape
        spec: list[tuple[str, tuple[int, ...], torch.dtype]] = [
            ("latent", (batch, int(c), int(t), int(h), int(w)), self.dtype),
        ]
        if include_hidden:
            spec.append(("hidden_states", (batch, int(self._seq_len), int(self.dim)), self.dtype))
        return spec

    def unpack_activation(self, payload: dict):
        return payload
