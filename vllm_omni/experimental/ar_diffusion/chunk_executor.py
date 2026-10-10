# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk pipeline slot loop: microbatch, chain P2P, noisy KV exchange."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    Cell,
    ChunkPlan,
    Inflight,
    can_admit,
    is_group_boundary,
    rank_work,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVState
from vllm_omni.experimental.ar_diffusion.phase_profile import (
    PhaseProfiler,
    bind_profiler,
    measure_forward,
    timed,
)

def activation_use_tensor_dict() -> bool:
    """Opt into legacy Gloo-metadata ``isend_tensor_dict`` activation path."""
    return os.environ.get("AR_DIFFUSION_ACT_USE_TENSOR_DICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pp_global_rank(pp_group: Any, group_rank: int) -> int:
    return int(pp_group.ranks[group_rank])


def _pp_group_for_tensor(pp_group: Any, tensor: torch.Tensor) -> Any:
    return pp_group.cpu_group if tensor.is_cpu else pp_group.device_group


def _stash_meta_from_tasks(src_tasks: tuple[tuple[str, Cell], ...]) -> tuple[str, int, int]:
    """Key peer stash without reading recv tensors (avoids sync before wait)."""
    _req, cell = src_tasks[0]
    chunk, step, _block = cell
    return ("", int(chunk), int(step))


def _activation_edges(
    *,
    topology: ChunkTopology,
    inflight: tuple[Any, ...],
    slot: int,
    schedule: Any,
) -> list[tuple[int, int, tuple[tuple[str, Cell], ...]]]:
    """Same edge list on every rank for this tick's activation P2P."""
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]] = []
    last_rank = topology.world - 1
    if topology.stages == 1:
        peer_tasks = rank_work(inflight, slot, last_rank)
        if _tasks_need_activation_send(peer_tasks, schedule):
            edges.append((last_rank, 0, peer_tasks))
    for src in range(last_rank):
        dst = src + 1
        src_tasks = rank_work(inflight, slot, src)
        if _tasks_need_activation_send(src_tasks, schedule):
            edges.append((src, dst, src_tasks))
    return edges


def _warmup_activation_p2p(pp_group: Any, *, world: int, rank: int, device: torch.device) -> None:
    """Ring ``batch_isend_irecv`` once so NCCL P2P channels exist before the hot path.

    Without this, the first unbatched/lazy P2P on a multi-rank group can host-block
    while creating 2-rank communicators (see ProcessGroupNCCL warnings).
    """
    if world <= 1 or not torch.cuda.is_available():
        return
    group = pp_group.device_group
    send_buf = torch.zeros(1, device=device, dtype=torch.float32)
    recv_buf = torch.empty(1, device=device, dtype=torch.float32)
    nxt = _pp_global_rank(pp_group, (rank + 1) % world)
    prv = _pp_global_rank(pp_group, (rank - 1 + world) % world)
    ops = [
        dist.P2POp(dist.isend, send_buf, nxt, group),
        dist.P2POp(dist.irecv, recv_buf, prv, group),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def _post_activations_ws(
    pp_group: Any,
    edges: list[tuple[int, int, tuple[tuple[str, Cell], ...]]],
    *,
    rank: int,
    output: Any | None,
    adapter: ChunkAdapter,
    topology: ChunkTopology,
    device: torch.device,
    profiler: PhaseProfiler,
) -> tuple[list[Any], list[torch.Tensor]]:
    """WaveServe-style activation: one ``batch_isend_irecv``, wait at consume.

    Every rank walks the same ``edges`` list. Local send/recv ops are posted in a
    single batch (no Gloo metadata). Recv ``Pending``-like wait happens in
    ``take_stashed_activation``; send keepalives stay alive via returned works.
    """
    ops: list[dist.P2POp] = []
    send_kept: list[torch.Tensor] = []
    recv_jobs: list[tuple[dict[str, torch.Tensor], tuple[str, int, int]]] = []
    group = pp_group.device_group

    for src, dst, src_tasks in edges:
        batch = max(1, len(src_tasks))
        include_hidden = not topology.is_stage_last(src)
        spec_list = adapter.activation_spec(batch=batch, include_hidden=include_hidden)
        if not spec_list:
            continue
        keys = tuple(k for k, _shape, _dtype in spec_list)
        if rank == src and output is not None:
            packed = adapter.pack_activation(output)
            peer = _pp_global_rank(pp_group, dst)
            for key in keys:
                tensor = packed[key]
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                send_kept.append(tensor)
                if tensor.is_cuda:
                    tensor.record_stream(torch.cuda.current_stream(tensor.device))
                ops.append(dist.P2POp(dist.isend, tensor, peer, group))
        elif rank == dst:
            peer = _pp_global_rank(pp_group, src)
            payload = {
                key: torch.empty(shape, dtype=dtype, device=device) for key, shape, dtype in spec_list
            }
            for key, _shape, _dtype in spec_list:
                ops.append(dist.P2POp(dist.irecv, payload[key], peer, group))
            recv_jobs.append((payload, _stash_meta_from_tasks(src_tasks)))

    t0 = time.perf_counter()
    works = list(dist.batch_isend_irecv(ops)) if ops else []
    profiler.add("act_batch_isend_irecv", time.perf_counter() - t0)
    profiler.add_total("act_batch_ops", len(ops))
    profiler.add_total("act_batch_works", len(works))

    # Match WaveServe: attach the batch work list to each received payload; the
    # consumer waits in take_stashed_activation (not in this tick's wait_outbound).
    for payload, meta in recv_jobs:
        adapter.stash_activation(payload, handles=list(works), meta=meta)

    return works, send_kept


def _prune_act_inflight(
    inflight: list[tuple[list[Any], list[torch.Tensor]]],
) -> list[tuple[list[Any], list[torch.Tensor]]]:
    """Drop completed activation send batches (WaveServe transport.inflight)."""
    kept: list[tuple[list[Any], list[torch.Tensor]]] = []
    for works, tensors in inflight:
        try:
            if works and all(getattr(w, "is_completed", lambda: False)() for w in works):
                continue
        except Exception:
            pass
        kept.append((works, tensors))
    return kept


def _profile_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return

# Backward-compatible alias.
ChunkStep = Cell

#: Wall time of each *active* slot (rank has work) on rank 0.
#: Experimental telemetry only; a harness clears it per request.
SLOT_TIMES: list[float] = []

#: End-to-end wall of one ``run_chunk_pipeline`` call on rank 0 (seconds).
PIPELINE_TIMES: list[float] = []


def _slot_times_file() -> str | None:
    return os.environ.get("AR_DIFFUSION_SLOT_TIMES_FILE") or None


def _pipeline_times_file() -> str | None:
    return os.environ.get("AR_DIFFUSION_PIPELINE_TIMES_FILE") or None


def _slot_telemetry_enabled() -> bool:
    """Opt-in only: sync+record costs a device barrier every active tick on rank 0."""
    return _slot_times_file() is not None


def _pipeline_telemetry_enabled() -> bool:
    return _pipeline_times_file() is not None


def _cuda_sync() -> None:
    try:
        import torch

        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
    except Exception:
        return


def _record_slot(delta: float) -> None:
    SLOT_TIMES.append(delta)
    path = _slot_times_file()
    if path:
        with open(path, "a") as fh:
            fh.write(f"{delta}\n")


def _record_pipeline(delta: float) -> None:
    PIPELINE_TIMES.append(delta)
    path = _pipeline_times_file()
    if path:
        with open(path, "a") as fh:
            fh.write(f"{delta}\n")


logger = init_logger(__name__)


def resolve_pp_rank_and_group() -> tuple[int, Any | None]:
    """Read the existing PP group; never creates a new process group."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_pipeline_parallel_rank,
            get_pp_group,
        )

        return int(get_pipeline_parallel_rank()), get_pp_group()
    except Exception:
        return 0, None


@dataclass(frozen=True)
class ChunkTopology:
    stages: int
    layer_groups: int

    @property
    def world(self) -> int:
        return self.stages * self.layer_groups

    def split_rank(self, rank: int) -> tuple[int, int]:
        if self.stages == 1:
            return 0, rank
        return rank // self.layer_groups, rank % self.layer_groups

    def is_stage_first(self, rank: int) -> bool:
        _, g = self.split_rank(rank)
        return g == 0

    def is_stage_last(self, rank: int) -> bool:
        _, g = self.split_rank(rank)
        return g == self.layer_groups - 1


@dataclass
class ChunkRunSpec:
    topology: ChunkTopology
    max_batch_size: int = 1
    rank: int = 0
    pp_group: Any | None = None


@dataclass(frozen=True)
class PendingAdmit:
    req: str
    plan: ChunkPlan
    chunk_tokens: int


@dataclass
class ARDiffusionChunkContext:
    spec: ChunkRunSpec
    kv: NoisyKVState
    inflight: list[Inflight] = field(default_factory=list)
    pending: list[PendingAdmit] = field(default_factory=list)

    def enqueue(self, req: str, plan: ChunkPlan, *, chunk_tokens: int) -> None:
        self.pending.append(PendingAdmit(req=req, plan=plan, chunk_tokens=chunk_tokens))

    def admit(self, req: str, plan: ChunkPlan, *, chunk_tokens: int, slot: int) -> bool:
        pending = tuple(self.inflight)
        if not can_admit(pending, slot, plan, max_batch_size=self.spec.max_batch_size):
            return False
        inflight_tokens = {self.kv._chunk_tokens[item.req] for item in self.inflight}
        if inflight_tokens and chunk_tokens not in inflight_tokens:
            return False
        self.kv.begin_request(req, plan, chunk_tokens=chunk_tokens, t0=slot)
        self.inflight.append(Inflight(req=req, t0=slot, plan=plan))
        self.kv.set_inflight(tuple(self.inflight))
        return True

    def drop(self, req: str) -> None:
        self.inflight = [item for item in self.inflight if item.req != req]
        self.pending = [item for item in self.pending if item.req != req]
        self.kv.end_request(req)
        self.kv.set_inflight(tuple(self.inflight))

    def admit_pending(self, slot: int) -> None:
        still: list[PendingAdmit] = []
        for item in self.pending:
            if still:
                still.append(item)
                continue
            if self.admit(item.req, item.plan, chunk_tokens=item.chunk_tokens, slot=slot):
                continue
            still.append(item)
        self.pending = still


class ChunkAdapter:
    """Model-side cell step. ``forward`` receives one block per request on this rank."""

    def forward(
        self,
        tasks: tuple[tuple[str, Cell], ...],
        kv_contexts: list[list[Any]],
        *,
        hidden: Any | None,
    ) -> Any:
        raise NotImplementedError

    def pack_activation(self, output: Any) -> dict:
        raise NotImplementedError

    def unpack_activation(self, payload: dict) -> Any:
        raise NotImplementedError

    def activation_spec(
        self, *, batch: int, include_hidden: bool
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]] | None:
        """Optional fixed recv schema for NCCL activation P2P.

        Return ``[(key, shape, dtype), ...]`` in send order, or ``None`` to
        fall back to ``isend_tensor_dict`` (Gloo metadata).
        """
        del batch, include_hidden
        return None

    def stash_activation(
        self,
        payload: dict,
        *,
        handles: list[Any] | None = None,
        meta: tuple[str, int, int] | None = None,
    ) -> None:
        """Optional: buffer a peer activation until the matching cell runs.

        ``handles`` are waited in ``take_stashed_activation`` (deferred recv).
        ``meta`` keys the stash without reading ``chunk``/``step`` tensors.
        """
        del payload, handles, meta

    def take_stashed_activation(self, req: str, chunk: int, step: int) -> Any | None:
        del req, chunk, step
        return None


def _wait(handles: list[Any]) -> None:
    for handle in handles:
        if handle is not None:
            handle.wait()


def _assert_i8(tasks: tuple[tuple[str, Cell], ...]) -> None:
    reqs = [req for req, _task in tasks]
    if len(reqs) != len(set(reqs)):
        raise RuntimeError(f"I8 violated: same request appears twice in one microbatch: {tasks}")


def _tasks_need_activation_send(
    tasks: tuple[tuple[str, Cell], ...],
    schedule: Any | None,
) -> bool:
    """PP activation leaves a rank only after its last local block of the group."""
    if not tasks or schedule is None:
        return bool(tasks)
    return any(is_group_boundary(cell, schedule) for _req, cell in tasks)


def run_chunk_pipeline(
    *,
    ctx: ARDiffusionChunkContext,
    adapter: ChunkAdapter,
    max_slots: int | None = None,
) -> None:
    """Drive inflight plans to completion on this rank (one cell per tick).

    Each tick waits on two disjoint sets only: the handles this rank posted
    (outbound not left dangling) and the inbound versions the *next* tick
    consumes. Activation P2P fires only on layer-group boundaries; mid-group
    cells keep hidden locally for the next block tick.
    """
    spec = ctx.spec
    kv = ctx.kv
    kv.bind_rank(spec.rank, spec.pp_group)
    pp = spec.pp_group
    logger.info(
        "[chunk_pipeline] start: stages=%d layer_groups=%d rank=%d inflight=%d pending=%d",
        spec.topology.stages,
        spec.topology.layer_groups,
        spec.rank,
        len(ctx.inflight),
        len(ctx.pending),
    )
    slot = 0
    limit = max_slots if max_slots is not None else 10**9
    pending_hidden: Any | None = None
    pipeline_started = time.perf_counter()
    profiler = PhaseProfiler.maybe(spec.rank)
    try:
        with bind_profiler(profiler):
            kv_stream = kv.uses_comm_stream
            prev_slot_for_evict: int | None = None
            # WaveServe-style: posted activation batches kept until complete / flush.
            act_inflight: list[tuple[list[Any], list[torch.Tensor]]] = []
            act_device = getattr(adapter, "device", None)
            if act_device is None and torch.cuda.is_available():
                act_device = torch.device("cuda", torch.cuda.current_device())
            elif act_device is None:
                act_device = torch.device("cpu")
            use_ws_act = (
                pp is not None
                and spec.topology.world > 1
                and not activation_use_tensor_dict()
                and adapter.activation_spec(batch=1, include_hidden=False) is not None
            )
            if use_ws_act:
                with timed(profiler, "act_p2p_warmup"):
                    _warmup_activation_p2p(
                        pp, world=spec.topology.world, rank=spec.rank, device=act_device
                    )
            while (ctx.inflight or ctx.pending) and slot < limit:
                slot_started = time.perf_counter()
                # Full-device sync kills comm/compute overlap; skip in stream mode.
                if profiler.enabled and not kv_stream and (ctx.inflight or ctx.pending):
                    t_stall = time.perf_counter()
                    _profile_cuda_sync()
                    profiler.add("stall_prev_gpu", time.perf_counter() - t_stall)
                ctx.admit_pending(slot)
                kv.set_inflight(tuple(ctx.inflight))
                tasks = rank_work(tuple(ctx.inflight), slot, spec.rank)
                _assert_i8(tasks)
                schedule = ctx.inflight[0].plan.schedule if ctx.inflight else None
                sample_gpu = bool(tasks) and profiler.note_active_tick()
                with timed(profiler, "prepare"):
                    contexts = kv.prepare(tasks) if tasks else []
                output = None
                if tasks:
                    # Dependency wait for peer activations happens inside adapter
                    # take_stashed_activation (WaveServe Pending.wait at consume).
                    output = measure_forward(
                        profiler,
                        lambda: adapter.forward(tasks, contexts, hidden=pending_hidden),
                        sample_gpu=sample_gpu,
                    )
                    kv.publish(tasks)
                    # Mid-group: keep hidden on this rank for the next local block.
                    if schedule is not None and not _tasks_need_activation_send(tasks, schedule):
                        pending_hidden = output
                    else:
                        pending_hidden = None
                # After forward is queued: finish prior KV isend, then evict last-use.
                if kv_stream and prev_slot_for_evict is not None:
                    with timed(profiler, "wait_outbound"):
                        kv.wait_sends()
                    with timed(profiler, "evict"):
                        kv.evict(prev_slot_for_evict)
                    prev_slot_for_evict = None
                comm_handles: list[Any] = []
                if pp is not None and spec.topology.world > 1 and schedule is not None:
                    edges = _activation_edges(
                        topology=spec.topology,
                        inflight=tuple(ctx.inflight),
                        slot=slot,
                        schedule=schedule,
                    )
                    with timed(profiler, "activation_post"):
                        if not use_ws_act:
                            for src, dst, src_tasks in edges:
                                if spec.rank == src and output is not None:
                                    t0 = time.perf_counter()
                                    handles = pp.isend_tensor_dict(
                                        adapter.pack_activation(output), dst=dst
                                    )
                                    profiler.add("act_isend_tensor_dict", time.perf_counter() - t0)
                                    comm_handles.extend(handles)
                                elif spec.rank == dst:
                                    t0 = time.perf_counter()
                                    recv, recv_handles, _ = pp.irecv_tensor_dict(src=src)
                                    profiler.add("act_irecv_tensor_dict", time.perf_counter() - t0)
                                    comm_handles.extend(recv_handles)
                                    adapter.stash_activation(
                                        recv, meta=_stash_meta_from_tasks(src_tasks)
                                    )
                        elif edges:
                            act_inflight = _prune_act_inflight(act_inflight)
                            works, kept = _post_activations_ws(
                                pp,
                                edges,
                                rank=spec.rank,
                                output=output,
                                adapter=adapter,
                                topology=spec.topology,
                                device=act_device,
                                profiler=profiler,
                            )
                            # Do not host-wait here — recv waits at take_stashed;
                            # send works stay in act_inflight (WaveServe transport).
                            if works:
                                act_inflight.append((works, kept))
                with timed(profiler, "kv_exchange"):
                    kv_handles = kv.exchange(slot)
                    if not kv_stream:
                        comm_handles.extend(kv_handles)
                    profiler.add_total("kv_handles", len(kv_handles) if not kv_stream else 0)
                # ① Outbound only (KV isend + legacy act). Recv wait is next tick's await_ready.
                with timed(profiler, "wait_outbound"):
                    _wait(comm_handles)
                still = []
                for item in ctx.inflight:
                    local = slot - item.t0
                    if local + 1 < item.plan.num_slots:
                        still.append(item)
                    else:
                        kv.end_request(item.req)
                ctx.inflight = still
                kv.set_inflight(tuple(ctx.inflight))
                if kv_stream:
                    # Defer evict(slot) to after next forward so isend overlaps compute.
                    prev_slot_for_evict = slot
                else:
                    with timed(profiler, "evict"):
                        kv.evict(slot)
                    # ② Original cell order (noisy_pp §2.5): await next tick's sources
                    # at end of this tick, immediately before the next prepare.
                    with timed(profiler, "await_ready"):
                        kv.await_ready(slot)
                # Hot path: do not cudaSynchronize every tick. Slot timing is
                # opt-in via AR_DIFFUSION_SLOT_TIMES_FILE (sync then required).
                if spec.rank == 0 and tasks and _slot_telemetry_enabled():
                    _cuda_sync()
                    _record_slot(time.perf_counter() - slot_started)
                slot += 1
            if kv_stream and prev_slot_for_evict is not None:
                with timed(profiler, "wait_outbound"):
                    kv.wait_sends()
                with timed(profiler, "evict"):
                    kv.evict(prev_slot_for_evict)
            # WaveServe transport.flush() equivalent for activation sends.
            if act_inflight:
                with timed(profiler, "act_flush"):
                    for works, _kept in act_inflight:
                        _wait(works)
                act_inflight = []
            with timed(profiler, "drain"):
                kv.drain()
        profiler.dump(
            extra={
                "stages": spec.topology.stages,
                "layer_groups": spec.topology.layer_groups,
                "slots": slot,
                "bytes_sent": getattr(kv, "bytes_sent", None),
                "bytes_received": getattr(kv, "bytes_received", None),
                "transfers_deduped": getattr(kv, "transfers_deduped", None),
                "pipeline_wall_s": time.perf_counter() - pipeline_started,
            }
        )
        if spec.rank == 0 and _pipeline_telemetry_enabled():
            _cuda_sync()
            _record_pipeline(time.perf_counter() - pipeline_started)
    except Exception:
        for req in {item.req for item in ctx.inflight} | {item.req for item in ctx.pending}:
            ctx.drop(req)
        raise


@contextmanager
def bind_ar_diffusion_chunk_context(owner: Any, ctx: ARDiffusionChunkContext) -> Iterator[ARDiffusionChunkContext]:
    prev = getattr(owner, "_ar_diffusion_chunk_context", None)
    owner._ar_diffusion_chunk_context = ctx
    try:
        yield ctx
    finally:
        owner._ar_diffusion_chunk_context = prev
