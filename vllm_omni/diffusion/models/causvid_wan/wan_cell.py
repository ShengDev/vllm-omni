# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One schedule cell on Wan DiT: ``embed → block → unembed`` (+ Euler)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.experimental.ar_diffusion.cell_executor import CellAdapter
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler


@dataclass
class WanCellStream:
    """Per-request state for Stream-Batch (``R>1``) on a shared transformer."""

    req_id: str
    sampler: FlowEuler
    prompt_embeds: torch.Tensor
    seed: int
    on_finished: Callable[[int, torch.Tensor], None] | None = None


class WanCell(CellAdapter):
    """Run one ``(chunk, step, block)`` cell; last stage applies FlowEuler and records latents.

    Single-request construction matches the historical API. Stream-Batch uses
    :meth:`from_streams` so several requests share one adapter / tick loop with
    per-req prompt / sampler / seed / finish callbacks.
    """

    def __init__(
        self,
        transformer: StageWanTransformer,
        *,
        sampler: FlowEuler,
        prompt_embeds: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        on_finished: Callable[[int, torch.Tensor], None] | None = None,
        req_id: str = "",
    ) -> None:
        self.transformer = transformer
        self.latent_shape = latent_shape
        self.device = device
        self.dtype = dtype
        self.num_denoise_steps = len(sampler.timesteps)
        self.finished: dict[str, dict[int, torch.Tensor]] = {}
        self._live: dict[str, dict[int, torch.Tensor]] = {}
        self._stash: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        # Optional completion tickets (CUDA Event or Work list) for async act recv.
        self._stash_ready: dict[tuple[str, int, int], list] = {}
        self._last_meta: tuple[str, int, int] | None = None
        self._timesteps = [torch.tensor(float(t), dtype=torch.float32) for t in sampler.timesteps]
        self._timestep_clean = torch.zeros((), dtype=torch.float32)
        self._streams: dict[str, WanCellStream] = {}
        self._bound_prompt_req: str | None = None
        # Legacy single-field aliases (first / only stream).
        self.sampler = sampler
        self.prompt_embeds = prompt_embeds
        self.seed = seed
        self.on_finished = on_finished
        self.add_stream(
            WanCellStream(
                req_id=req_id,
                sampler=sampler,
                prompt_embeds=prompt_embeds,
                seed=seed,
                on_finished=on_finished,
            )
        )

    @classmethod
    def from_streams(
        cls,
        transformer: StageWanTransformer,
        streams: list[WanCellStream],
        *,
        latent_shape: tuple[int, int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> WanCell:
        if not streams:
            raise ValueError("from_streams requires at least one WanCellStream")
        first = streams[0]
        cell = cls(
            transformer,
            sampler=first.sampler,
            prompt_embeds=first.prompt_embeds,
            latent_shape=latent_shape,
            seed=first.seed,
            device=device,
            dtype=dtype,
            on_finished=first.on_finished,
            req_id=first.req_id,
        )
        for stream in streams[1:]:
            if len(stream.sampler.timesteps) != cell.num_denoise_steps:
                raise ValueError("Stream-Batch requires equal num_denoise_steps across requests")
            cell.add_stream(stream)
        return cell

    def add_stream(self, stream: WanCellStream) -> None:
        if stream.req_id in self._streams and self._streams[stream.req_id] is not stream:
            raise ValueError(f"duplicate Stream-Batch req_id {stream.req_id!r}")
        self._streams[stream.req_id] = stream
        if len(self._streams) == 1:
            self.sampler = stream.sampler
            self.prompt_embeds = stream.prompt_embeds
            self.seed = stream.seed
            self.on_finished = stream.on_finished
            self.transformer.prepare_cell(stream.prompt_embeds)
            self._bound_prompt_req = stream.req_id

    def _stream_for(self, req: str) -> WanCellStream:
        if req in self._streams:
            return self._streams[req]
        if len(self._streams) == 1:
            return next(iter(self._streams.values()))
        raise KeyError(f"no WanCell stream for req {req!r}; known={list(self._streams)}")

    def _ensure_prompt(self, req: str) -> WanCellStream:
        stream = self._stream_for(req)
        if self._bound_prompt_req != stream.req_id:
            self.transformer.prepare_cell(stream.prompt_embeds)
            self._bound_prompt_req = stream.req_id
        return stream

    def _init_noise(self, req: str, chunk: int) -> torch.Tensor:
        stream = self._stream_for(req)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(stream.seed * 1_000_003 + chunk * 4096)
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
        tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor) and k not in ("chunk", "step")}
        key = meta or self._last_meta
        if key is None:
            raise RuntimeError("stash_activation needs meta")
        self._stash[key] = tensors
        if handles:
            self._stash_ready[key] = list(handles)

    def _wait_stash_ready(self, key: tuple[str, int, int]) -> None:
        ready = self._stash_ready.pop(key, None)
        if not ready:
            return
        for handle in ready:
            if handle is None:
                continue
            if isinstance(handle, torch.cuda.Event):
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).wait_event(handle)
                else:
                    handle.synchronize()
            else:
                handle.wait()

    def take_stashed_activation(self, req: str, chunk: int, step: int):
        for key in ((req, chunk, step), ("", chunk, step)):
            tensors = self._stash.pop(key, None)
            if tensors is not None:
                self._wait_stash_ready(key)
                return tensors
        return None

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
            stored = live.get(chunk)
            latent = self._init_noise(req, chunk) if stored is None else stored
        return latent, tokens

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        outs: list[dict[str, torch.Tensor]] = []
        tf = self.transformer
        for i, (req, cell) in enumerate(tasks):
            chunk, step, block = cell
            stream = self._ensure_prompt(req)
            self._last_meta = (req, chunk, step)
            ctx_list = kv_contexts[i] if i < len(kv_contexts) else None
            kv_ctx = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            live = self._live.setdefault(req, {})
            latent, tokens = self._resolve_latent_hidden(
                req=req, chunk=chunk, step=step, index=i, n_tasks=len(tasks), hidden=hidden, live=live
            )
            t = self._timesteps[step] if step < self.num_denoise_steps else self._timestep_clean
            step_key = step if step < self.num_denoise_steps else self.num_denoise_steps

            if block == tf.start_layer and tf.is_layer_group_first:
                tokens = tf.embed(latent)
            elif tokens is None:
                raise RuntimeError(f"cell {(chunk, step, block)} missing hidden_states")
            tokens = tf.block(block, tokens, chunk=chunk, step=step_key, latent=latent, timestep=t, kv_ctx=kv_ctx)
            if not (block == tf.end_layer - 1 and tf.is_layer_group_last):
                live[chunk] = latent
                outs.append({"latent": latent, "hidden_states": tokens})
                continue
            pred = tf.unembed(tokens, chunk=chunk, step=step_key, latent=latent, timestep=t)
            if step < self.num_denoise_steps:
                latent = stream.sampler.advance(pred, latent, step)
                if step == self.num_denoise_steps - 1:
                    self.finished.setdefault(req, {})[chunk] = latent.detach()
                    if stream.on_finished is not None:
                        stream.on_finished(chunk, latent)
            else:
                self.finished.setdefault(req, {})[chunk] = latent.detach()
            live[chunk] = latent
            outs.append({"latent": latent})
        if len(outs) == 1:
            return outs[0]
        # Stream-Batch pad: all outs in one tick must share keys (K=1 ranks are uniform).
        keys = outs[0].keys()
        if any(o.keys() != keys for o in outs):
            raise RuntimeError(
                "Stream-Batch microbatch mixed activation schemas; "
                "use K=1 or same-phase cells on a rank"
            )
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
            seq = self.transformer.seq_len_for_latent((batch, int(c), int(t), int(h), int(w)))
            spec.append(("hidden_states", (batch, seq, int(self.transformer.dim)), self.dtype))
        return spec

    def unpack_activation(self, payload: dict):
        return payload
