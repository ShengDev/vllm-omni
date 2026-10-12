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
    ChunkPlan,
    ChunkStep,
    Inflight,
    can_admit,
    rank_work,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    NoisyKVState,
    _pp_global_rank,
    _pp_group_for_tensor,
)

#: Wall time of each *active* slot (rank has work) on rank 0.
#: Experimental telemetry only; a harness clears it per request.
SLOT_TIMES: list[float] = []

#: End-to-end wall of one ``run_chunk_pipeline`` call on rank 0 (seconds).
PIPELINE_TIMES: list[float] = []


def _slot_times_file() -> str | None:
    return os.environ.get("AR_DIFFUSION_SLOT_TIMES_FILE") or None


def _pipeline_times_file() -> str | None:
    return os.environ.get("AR_DIFFUSION_PIPELINE_TIMES_FILE") or None


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


@dataclass(frozen=True)
class ChunkParallelLayout:
    """DiT chunk-PP ranks plus an optional dedicated VAE decoder rank.

    When ``decoder_rank`` is set, ``pipeline_parallel_size = dit_world + 1`` and
    the last PP rank owns only VAE decode. DiT KV/activation P2P stays inside
    ``topology.world``; the decoder never joins ``run_chunk_pipeline``.
    """

    topology: ChunkTopology
    decoder_rank: int | None = None

    @property
    def dit_world(self) -> int:
        return self.topology.world

    @property
    def dedicated_decoder(self) -> bool:
        return self.decoder_rank is not None

    def is_decoder_rank(self, rank: int) -> bool:
        return self.decoder_rank is not None and rank == self.decoder_rank

    def is_dit_rank(self, rank: int) -> bool:
        return 0 <= rank < self.dit_world


def resolve_chunk_parallel_layout(
    *,
    stages: int,
    pp_world: int,
    dedicated_decoder: bool = False,
) -> ChunkParallelLayout:
    """Map deploy PP size onto DiT topology (+ optional trailing decoder rank)."""
    if stages < 1:
        raise ValueError(f"stage_parallel_size must be >= 1, got {stages}")
    if pp_world < 1:
        raise ValueError(f"pipeline_parallel_size must be >= 1, got {pp_world}")
    if dedicated_decoder:
        dit_world = pp_world - 1
        if dit_world < stages:
            raise ValueError(
                "dedicated_decoder requires pipeline_parallel_size >= stage_parallel_size + 1; "
                f"got pipeline_parallel_size={pp_world}, stage_parallel_size={stages}"
            )
        if dit_world % stages != 0:
            raise ValueError(
                "with dedicated_decoder, (pipeline_parallel_size - 1) must be divisible by "
                f"stage_parallel_size; got dit_world={dit_world}, stages={stages}"
            )
        layer_groups = dit_world // stages
        return ChunkParallelLayout(
            topology=ChunkTopology(stages=stages, layer_groups=layer_groups),
            decoder_rank=dit_world,
        )
    if pp_world % stages != 0:
        raise ValueError(f"pipeline_parallel_size={pp_world} must be divisible by stage_parallel_size={stages}")
    return ChunkParallelLayout(
        topology=ChunkTopology(stages=stages, layer_groups=pp_world // stages),
        decoder_rank=None,
    )


@dataclass(frozen=True)
class DitProcessGroup:
    """NCCL/Gloo subgroup covering only DiT ranks (excludes dedicated decoder).

    Activation and NoisyKV ``batch_isend_irecv`` must run on this group when a
    dedicated decoder shares the PP world: PyTorch requires every member of the
    P2P ``group`` to enter the *first* ``batch_isend_irecv``, and idle decoder
    ranks never join the chunk loop.
    """

    ranks: list[int]
    device_group: Any
    cpu_group: Any

    @property
    def world_size(self) -> int:
        return len(self.ranks)


def make_dit_process_group(pp_group: Any, dit_world: int) -> DitProcessGroup | None:
    """Create DiT-only subgroups; **all** PP ranks must call this collectively."""
    if pp_group is None or dit_world < 1:
        return None
    ranks = list(getattr(pp_group, "ranks", []) or [])
    if not ranks or dit_world > len(ranks):
        raise ValueError(f"dit_world={dit_world} incompatible with pp ranks={ranks}")
    if dit_world == len(ranks):
        # No dedicated decoder — reuse the existing PP coordinator as-is.
        return None
    dit_ranks = ranks[:dit_world]
    backend = dist.get_backend(getattr(pp_group, "device_group", None) or pp_group)
    device_group = dist.new_group(ranks=dit_ranks, backend=backend)
    cpu_group = dist.new_group(ranks=dit_ranks, backend="gloo")
    return DitProcessGroup(ranks=dit_ranks, device_group=device_group, cpu_group=cpu_group)


def prime_dit_process_group(dit: DitProcessGroup | None, rank: int) -> None:
    """Barrier on the DiT device group so later subset P2P is well-defined."""
    if dit is None or rank not in dit.ranks:
        return
    dist.barrier(group=dit.device_group)


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
    """Model-side chunk step. ``forward`` receives the microbatch on this rank."""

    def forward(
        self,
        tasks: tuple[tuple[str, ChunkStep], ...],
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
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        """Fixed activation shapes for ``batch_isend_irecv`` (required when PP>1)."""
        raise NotImplementedError


def _wait(handles: list[Any]) -> None:
    for handle in handles:
        if handle is not None:
            handle.wait()


def _assert_i8(tasks: tuple[tuple[str, ChunkStep], ...]) -> None:
    reqs = [req for req, _task in tasks]
    if len(reqs) != len(set(reqs)):
        raise RuntimeError(f"I8 violated: same request appears twice in one microbatch: {tasks}")


def _act_peers(rank: int, world: int, stages: int) -> tuple[int | None, int | None]:
    """Precomputed activation ring peers for this rank (no per-slot topology walk).

    Chain ``r → r+1``; when ``stages==1`` also wrap ``last → 0``.
    """
    send_to = rank + 1 if rank < world - 1 else (0 if stages == 1 else None)
    recv_from = rank - 1 if rank > 0 else (world - 1 if stages == 1 else None)
    return send_to, recv_from


def _post_activation_batch(
    pp: Any,
    *,
    rank: int,
    send_to: int | None,
    recv_from: int | None,
    do_send: bool,
    do_recv: bool,
    output: Any | None,
    adapter: ChunkAdapter,
    topology: ChunkTopology,
    batch: int,
    device: torch.device,
) -> tuple[list[Any], Any | None]:
    """One ``batch_isend_irecv`` for this rank's activation edge(s) (no Gloo metadata)."""
    send_spec = (
        adapter.activation_spec(batch=batch, include_hidden=not topology.is_stage_last(rank))
        if do_send
        else []
    )
    recv_spec = (
        adapter.activation_spec(batch=batch, include_hidden=not topology.is_stage_last(recv_from or 0))
        if do_recv and recv_from is not None
        else []
    )
    if do_send and not send_spec:
        raise RuntimeError("activation send requires adapter.activation_spec()")
    if do_recv and not recv_spec:
        raise RuntimeError("activation recv requires adapter.activation_spec()")

    ops: list[dist.P2POp] = []
    send_kept: list[torch.Tensor] = []
    pending_hidden = None

    if do_send and send_to is not None and output is not None:
        packed = adapter.pack_activation(output)
        peer = _pp_global_rank(pp, send_to)
        for key, _shape, _dtype in send_spec:
            tensor = packed[key]
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            send_kept.append(tensor)
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            ops.append(dist.P2POp(dist.isend, tensor, peer, _pp_group_for_tensor(pp, tensor)))

    if do_recv and recv_from is not None:
        peer = _pp_global_rank(pp, recv_from)
        payload = {key: torch.empty(shape, dtype=dtype, device=device) for key, shape, dtype in recv_spec}
        for key, _shape, _dtype in recv_spec:
            tensor = payload[key]
            ops.append(dist.P2POp(dist.irecv, tensor, peer, _pp_group_for_tensor(pp, tensor)))
        pending_hidden = adapter.unpack_activation(payload)

    works = list(dist.batch_isend_irecv(ops)) if ops else []
    del send_kept  # kept alive via ops/tensor refs until works wait
    return works, pending_hidden


def run_chunk_pipeline(
    *,
    ctx: ARDiffusionChunkContext,
    adapter: ChunkAdapter,
    max_slots: int | None = None,
) -> None:
    """Drive inflight plans to completion on this rank.

    Each slot waits on two disjoint sets only: the handles this rank posted
    (outbound not left dangling) and the inbound versions the *next* slot
    consumes. Everything else keeps overlapping with the next forward.
    """
    spec = ctx.spec
    kv = ctx.kv
    kv.bind_rank(spec.rank, spec.pp_group)
    pp = spec.pp_group
    world = spec.topology.world
    # Activation topology is static for the run — compute peers once.
    act_send_to, act_recv_from = _act_peers(spec.rank, world, spec.topology.stages)
    logger.info(
        "[chunk_pipeline] start: stages=%d layer_groups=%d rank=%d inflight=%d pending=%d "
        "act_send_to=%s act_recv_from=%s",
        spec.topology.stages,
        spec.topology.layer_groups,
        spec.rank,
        len(ctx.inflight),
        len(ctx.pending),
        act_send_to,
        act_recv_from,
    )
    slot = 0
    limit = max_slots if max_slots is not None else 10**9
    pending_hidden: Any | None = None
    pipeline_started = time.perf_counter()
    try:
        while (ctx.inflight or ctx.pending) and slot < limit:
            slot_started = time.perf_counter()
            ctx.admit_pending(slot)
            kv.set_inflight(tuple(ctx.inflight))
            tasks = rank_work(tuple(ctx.inflight), slot, spec.rank)
            _assert_i8(tasks)
            contexts = kv.prepare(tasks) if tasks else []
            output = None
            if tasks:
                output = adapter.forward(tasks, contexts, hidden=pending_hidden)
                kv.publish(tasks)
            comm_handles: list[Any] = []
            if pp is not None and world > 1:
                inflight_t = tuple(ctx.inflight)
                do_send = bool(tasks) and act_send_to is not None
                do_recv = act_recv_from is not None and bool(
                    rank_work(inflight_t, slot, act_recv_from)
                )
                if do_send or do_recv:
                    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
                    # Prefer a tensor device from output when present.
                    if output is not None and isinstance(output, dict):
                        for v in output.values():
                            if isinstance(v, torch.Tensor):
                                device = v.device
                                break
                    act_works, recv_hidden = _post_activation_batch(
                        pp,
                        rank=spec.rank,
                        send_to=act_send_to,
                        recv_from=act_recv_from,
                        do_send=do_send,
                        do_recv=do_recv,
                        output=output,
                        adapter=adapter,
                        topology=spec.topology,
                        batch=max(1, len(tasks) if tasks else 1),
                        device=device,
                    )
                    comm_handles.extend(act_works)
                    if do_recv:
                        pending_hidden = recv_hidden
            comm_handles.extend(kv.exchange(slot))
            # ① Outbound not left dangling: only this rank's own posted handles.
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
            # ② Next slot's sources: wait only the inbound versions slot+1 reads.
            kv.evict(slot)
            kv.await_ready(slot)
            if spec.rank == 0 and tasks:
                # Active slots only — idle bubbles would dominate SERIAL medians.
                _cuda_sync()
                _record_slot(time.perf_counter() - slot_started)
            slot += 1
        # Teardown: land anything the narrowed waits skipped.
        kv.drain()
        if spec.rank == 0:
            _cuda_sync()
            _record_pipeline(time.perf_counter() - pipeline_started)
    except Exception:
        # Fail-closed: release VersionPool slots so later requests are not stranded.
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
