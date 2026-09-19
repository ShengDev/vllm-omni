# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk scheduling over existing layer-split PP stages.

Layer weights are partitioned at model build time. This module schedules
temporal ``(chunk, step)`` work, optional KV history, and stage
communication. KV source policy (latest/clean) is decoupled from the slot
schedule, and every published version carries a producer identity so
dual-expert checkpoints keep independent per-tower caches.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

ChunkStep = tuple[int, int]

# Request extra_args keys shared by the engine dummy request, the pipeline,
# the benchmark client and tests. Keep one source of truth; the engine dummy
# request historically sent "chunk_lag" while the pipeline read "chunk_gap",
# so the canonical key is now CHUNK_GAP_KEY everywhere.
CHUNK_SCHEDULE_KEY = "chunk_schedule"
CHUNK_FRAMES_KEY = "chunk_frames"
CHUNK_COND_FRAMES_KEY = "chunk_cond_frames"
CHUNK_GAP_KEY = "chunk_gap"
CHUNK_CONDITIONING_KEY = "chunk_conditioning"
KV_HISTORY_CHUNKS_KEY = "kv_history_chunks"
KV_SOURCE_POLICY_KEY = "kv_source_policy"
COLLECT_PP_METRICS_KEY = "collect_pp_metrics"
EXPERIMENT_ITERATION_KEY = "experiment_iteration"

# KV source policies, decoupled from the slot schedule:
# - "latest": each chunk reads the highest finished version of its history
#   chunks (``plan_latest_kv_sources``). Valid under both schedules; serial
#   + latest is the replay control that isolates the schedule's contribution.
# - "clean": each chunk reads only the clean pass of its history chunks
#   (``plan_clean_kv_sources``). Requires the serial schedule, whose strict
#   ordering guarantees the clean pass has run; this is the Self Forcing
#   semantics aligned with the paper.
KV_SOURCE_POLICIES = ("latest", "clean")


def plan_chunk_pipeline(
    chunks: int,
    world: int,
    schedule: str = "stepwise",
    *,
    kv: bool = False,
    num_denoise_steps: int = 3,
) -> list[tuple[ChunkStep | None, ...]]:
    """Plan per-slot work for each PP stage.

    Returns slots of length ``world``. A job enters stage 0 and advances one
    stage per slot; completion on the last stage unlocks dependents.
    ``serial`` runs one chunk at a time; ``stepwise`` interleaves waves of
    ``world`` chunks. With ``kv``, each chunk includes one extra clean step
    (step == num_denoise_steps): it publishes KV and does not denoise.
    """
    if chunks < 1 or world < 1:
        raise ValueError("Chunk pipeline requires positive chunks and pipeline_parallel_size >= 1")
    if num_denoise_steps < 1:
        raise ValueError("Chunk pipeline requires a positive number of denoise steps")
    steps = num_denoise_steps + 1 if kv else num_denoise_steps
    if schedule == "serial":
        jobs = [(chunk, step) for chunk in range(chunks) for step in range(steps)]
    elif schedule == "stepwise":
        jobs = [
            (chunk, step)
            for first in range(0, chunks, world)
            for step in range(steps)
            for chunk in range(first, min(first + world, chunks))
        ]
    else:
        raise ValueError("Chunk schedule must be 'serial' or 'stepwise'")

    slots = []
    completed = set()
    carry = [None] * (world - 1)
    cursor = 0
    while cursor < len(jobs) or any(task is not None for task in carry):
        launched = None
        if cursor < len(jobs) and not (schedule == "serial" and any(task is not None for task in carry)):
            chunk, step = jobs[cursor]
            dependencies = {(chunk, step - 1)} if step else set()
            if dependencies <= completed:
                launched = jobs[cursor]
                cursor += 1
        slot = (launched, *carry)
        if all(task is None for task in slot):
            raise RuntimeError("Chunk schedule cannot satisfy the next task's dependencies")
        slots.append(slot)
        if slot[-1] is not None:
            completed.add(slot[-1])
        carry = list(slot[:-1])
    return slots


def plan_latest_kv_sources(slots, history_chunks: int):
    """Freeze latest completed versions BEFORE a slot, separately per layer stage.

    The history of each chunk concatenates the highest finished denoise or
    clean step of its history chunks as of the slot's position in the plan.
    Valid under both schedules: under stepwise this is the algorithm as
    designed, under serial it is the replay control (identical versions to
    stepwise, no parallelism). Every history chunk in the window must have a
    finished version on the reading rank. The extra final step is a forward
    at t=0 on the clean sample; it does not denoise.
    """
    if history_chunks < 1:
        raise ValueError("KV history must contain at least one chunk")
    latest = [{} for _ in slots[0]]
    sources = [{} for _ in slots[0]]
    for tasks in slots:
        for rank, task in enumerate(tasks):
            if task is not None:
                chunk, _ = task
                window = range(max(0, chunk - history_chunks), chunk)
                missing = [c for c in window if c not in latest[rank]]
                if missing:
                    raise RuntimeError(
                        f"Task {task} on rank {rank} is scheduled before history chunks {missing} "
                        "have a finished version on that rank"
                    )
                sources[rank][task] = tuple((c, latest[rank][c]) for c in window)
        for rank, task in enumerate(tasks):
            if task is not None:
                chunk, step = task
                latest[rank][chunk] = step
    return sources


def plan_kv_last_use(slots, sources_by_rank):
    """Last slot on each rank that still references a published KV version."""
    last_use = [{} for _ in slots[0]]
    for slot_idx, tasks in enumerate(slots):
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            last_use[rank][task] = max(last_use[rank].get(task, -1), slot_idx)
            for source in sources_by_rank[rank][task]:
                last_use[rank][source] = max(last_use[rank].get(source, -1), slot_idx)
    return last_use


def evict_kv_versions(layers: dict, versions) -> None:
    """Remove KV versions from the per-layer cache; drop empty layer entries."""
    empty = []
    for layer, cache in layers.items():
        for version in versions:
            cache.pop(version, None)
        if not cache:
            empty.append(layer)
    for layer in empty:
        del layers[layer]


def plan_clean_kv_sources(slots, history_chunks: int, clean_step: int):
    """Serial Self Forcing: each chunk concatenates predecessors' clean KV only.

    Serial ordering guarantees every chunk in the window has run its clean pass
    (task (chunk, clean_step)) before the reading chunk starts, so the sources
    do not depend on slot completion. The clean pass publishes its KV and does
    not denoise.
    """
    if history_chunks < 1:
        raise ValueError("KV history must contain at least one chunk")
    if clean_step < 0:
        raise ValueError("Clean KV step must be non-negative")
    sources = [{} for _ in slots[0]]
    for tasks in slots:
        for rank, task in enumerate(tasks):
            if task is None:
                continue
            chunk, _ = task
            start = max(0, chunk - history_chunks)
            sources[rank][task] = tuple((c, clean_step) for c in range(start, chunk))
    return sources


@dataclass
class ChunkKVContext:
    """Request-local, stage-local cache of post-normalization/post-RoPE K and V.

    Every published version carries the producer identity (which tower ran
    the forward). Readers only ever consume versions produced by their own
    tower, mirroring the reference implementation's two independent per-tower
    caches; the planner-level sources are tower-agnostic and are resolved
    against the reading tower's producer here.
    """

    layers: dict
    task: ChunkStep
    sources: tuple[ChunkStep, ...]
    producer: str | None = None

    def _cache_for(self, layer: int) -> dict:
        return self.layers.setdefault(layer, {})

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        cache = self._cache_for(layer)
        store_key = (self.task, self.producer)
        if store_key in cache:
            raise RuntimeError(
                f"Duplicate KV publication for layer {layer}, task {self.task}, producer {self.producer}"
            )
        history = [cache[(source, self.producer)] for source in self.sources]
        cache[store_key] = (key.contiguous(), value.contiguous())
        if not history:
            return key, value
        return (
            torch.cat([pair[0] for pair in history] + [key], dim=1),
            torch.cat([pair[1] for pair in history] + [value], dim=1),
        )


def _tensor_bytes(payload: dict[str, torch.Tensor] | None) -> int:
    return sum(value.numel() * value.element_size() for value in payload.values()) if payload else 0


def _kv_retained_bytes(kv_layers: dict) -> int:
    """Logical bytes of the K/V tensors currently cached (numel * element_size),
    not allocator reservations; the cache stores post-RoPE tensors per layer."""
    total = 0
    for cache in kv_layers.values():
        for key, value in cache.values():
            total += key.numel() * key.element_size() + value.numel() * value.element_size()
    return total


@torch.inference_mode()
def run_chunk_pipeline(
    *,
    predict_noise: Callable,
    update_sample: Callable,
    timesteps: torch.Tensor,
    shape: tuple[int, ...],
    chunks: int,
    seed: int,
    device: torch.device,
    schedule: str = "stepwise",
    layer_range: tuple[int, int] | None = None,
    initial_latents: torch.Tensor | None = None,
    step_noises: list[torch.Tensor] | None = None,
    kv_history_chunks: int | None = None,
    kv_source_policy: str = "latest",
    kv_producers: tuple[str, ...] = ("single",),
    kv_reader_producer: Callable | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run the chunk timetable on this PP rank.

    All ranks share the same slots; rank ``r`` executes ``tasks[r]``.
    ``predict_noise`` accepts (model_input, timestep, temporal_offset,
    step_idx, producer, intermediate_tensors=None, kv_context=None) and
    returns ``IntermediateTensors`` on non-last stages and a noise tensor
    on the last stage. ``update_sample`` performs the last-stage latent
    update (the shared native/chunk contract). Per slot, communicate
    last→0 feedback, then stage i→i+1 activations.

    ``kv_producers`` names the KV producer identities (towers): the clean
    pass runs once per producer and each version is stored under its
    producer; every denoise forward consumes only history produced by its
    own tower (``kv_reader_producer(timestep) -> producer``). KV mode
    requires shared ``initial_latents`` and one re-noise tensor per step
    transition.
    """
    pp = get_pp_group()
    rank, world = pp.rank_in_group, pp.world_size
    last_rank = world - 1
    kv = kv_history_chunks is not None
    num_denoise_steps = len(timesteps)
    last_denoise = num_denoise_steps - 1
    slots = plan_chunk_pipeline(chunks, world, schedule, kv=kv, num_denoise_steps=num_denoise_steps)
    kv_layers = {}
    kv_sources = None
    kv_remaining = None
    kv_live = 0
    kv_peak_live = 0
    kv_evicted = 0
    kv_peak_retained_bytes = 0
    kv_last_use = None
    kv_peak_versions = 0
    if kv:
        if kv_source_policy not in KV_SOURCE_POLICIES:
            raise ValueError(f"KV source policy must be one of {KV_SOURCE_POLICIES}")
        if kv_source_policy == "clean" and schedule != "serial":
            raise ValueError(
                "The clean KV source policy requires the serial schedule: only its strict "
                "ordering guarantees every history chunk has run its clean pass"
            )
        # The "latest" policy always freezes sources against the stepwise
        # reference plan: under stepwise that is the algorithm as designed,
        # under serial it makes the run a replay control whose attention
        # history is identical to the stepwise schedule's. The "clean" policy
        # reads the serial plan's own completion order instead.
        if kv_source_policy == "clean":
            sources_by_rank = plan_clean_kv_sources(slots, kv_history_chunks, num_denoise_steps)
        else:
            reference_slots = slots
            if schedule != "stepwise":
                reference_slots = plan_chunk_pipeline(
                    chunks, world, "stepwise", kv=True, num_denoise_steps=num_denoise_steps
                )
            sources_by_rank = plan_latest_kv_sources(reference_slots, kv_history_chunks)
        kv_sources = sources_by_rank[rank]
        kv_last_use = plan_kv_last_use(slots, sources_by_rank)[rank]
        if initial_latents is None or step_noises is None or len(step_noises) != last_denoise:
            raise ValueError(
                "KV execution requires the shared full-video initial sample "
                "and one re-noising tensor per step transition"
            )
        # Each (task, producer) version is consumed by the later same-rank
        # tasks that list it in their frozen sources, one consumption per
        # producer. Zero-consumer versions free immediately.
        kv_remaining = {(task, producer): 0 for task in kv_sources for producer in kv_producers}
        for sources in kv_sources.values():
            for source in sources:
                for producer in kv_producers:
                    kv_remaining[(source, producer)] += 1
    if num_denoise_steps < 1:
        raise ValueError("Chunk pipeline requires a positive number of denoise steps")
    t_l = shape[2]
    current, generators, clean_chunks = {}, {}, {}
    records, slot_records = [], []
    # CUDA events are resolved after the request completes. Resolving each
    # event in the hot loop would make the CPU wait for every stage forward.
    pending_timings = []
    comm_ms = 0.0
    activation_bytes = sample_bytes = feedback_bytes = 0
    stage_input = None
    trace_enabled = os.environ.get("VLLM_OMNI_CHUNK_PP_TRACE") == "1"

    def trace_scope(name: str):
        return torch.profiler.record_function(name) if trace_enabled else nullcontext()

    def accept_feedback(task, payload):
        chunk, step = task
        current[chunk] = payload["latents"]
        if step == last_denoise:
            clean_chunks[chunk] = current[chunk]

    def initialize_chunk(chunk):
        if kv:
            if rank == 0:
                current[chunk] = initial_latents[:, :, chunk * t_l : (chunk + 1) * t_l].contiguous()
            return
        generator = torch.Generator(device=device).manual_seed(seed + chunk * 100003)
        generators[chunk] = generator
        initial = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        if rank == 0:
            current[chunk] = initial
        # Rank one consumes the same initialization draw; its exact sample
        # arrives with model_input, and this generator owns the later draws.

    def barrier():
        if world > 1:
            dist.barrier(group=pp.device_group)

    def release_kv(task, published: tuple[str, ...]):
        nonlocal kv_live, kv_peak_live, kv_evicted, kv_peak_retained_bytes
        # Sample both peaks at the residency maximum, before any release: the
        # task just published its version(s), so bytes mirror the live count.
        kv_live += len(published)
        kv_peak_live = max(kv_peak_live, kv_live)
        kv_peak_retained_bytes = max(kv_peak_retained_bytes, _kv_retained_bytes(kv_layers))
        for source in kv_sources[task]:
            for producer in kv_producers:
                remaining_key = (source, producer)
                kv_remaining[remaining_key] -= 1
                if kv_remaining[remaining_key] == 0:
                    kv_live -= 1
                    kv_evicted += 1
                    for layer_cache in kv_layers.values():
                        layer_cache.pop(remaining_key, None)
        for producer in published:
            remaining_key = (task, producer)
            if kv_remaining[remaining_key] == 0:
                kv_live -= 1
                kv_evicted += 1
                for layer_cache in kv_layers.values():
                    layer_cache.pop(remaining_key, None)

    # Keep metrics collection out of the timed request path. ``timesteps`` is
    # resident on the accelerator, so converting it in a per-slot record would
    # otherwise introduce a device-to-host synchronization for every forward.
    timestep_values = tuple(float(value) for value in timesteps.detach().cpu().flatten().tolist())
    torch.accelerator.synchronize(device)
    barrier()
    torch.accelerator.reset_peak_memory_stats(device)
    # All chunk forwards and P2P waits are issued on this one stream, so
    # slots execute in issue order and a per-slot Work.wait() is enough to
    # keep consumption ordered; the per-slot CUDA events only timestamp the
    # forwards (Event.record records a time, it synchronizes nothing).
    stream = torch.cuda.current_stream(device)
    origin = torch.cuda.Event(enable_timing=True)
    origin.record(stream)
    start_wall = time.perf_counter()
    for slot_idx, tasks in enumerate(slots):
        task = tasks[rank]
        forward_payload = feedback_payload = None
        record = None
        if task is not None:
            chunk, step = task
            clean_pass = kv and step == num_denoise_steps
            t = timesteps.new_zeros(()) if clean_pass else timesteps[step]
            if step == 0:
                initialize_chunk(chunk)
            offset = chunk * t_l
            intermediate = None
            if rank == 0:
                model_input = current[chunk]
            else:
                model_input = stage_input["model_input"]
                intermediate = IntermediateTensors(
                    {key: value for key, value in stage_input.items() if key != "model_input"}
                )
            first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            first.record(stream)
            if kv:
                # Per the reference contract, the clean/context pass runs once
                # per tower and each tower stores its own version; a denoise
                # forward consumes history produced by its own tower only.
                reader = kv_reader_producer(timestep_values[step] if not clean_pass else 0.0) if kv_reader_producer else kv_producers[0]
                publishers = kv_producers if clean_pass else (reader,)
                kwargs = {"intermediate_tensors": intermediate}
                prediction = None
                for producer in publishers:
                    kv_context = ChunkKVContext(kv_layers, task, kv_sources[task], producer=producer)
                    kwargs["kv_context"] = kv_context
                    with trace_scope(f"chunk_pp.forward.slot{slot_idx}.rank{rank}.chunk{chunk}.step{step}.{producer}"):
                        prediction = predict_noise(model_input, t.expand(shape[0]), offset, step, producer, **kwargs)
            else:
                prediction = predict_noise(
                    model_input, t.expand(shape[0]), offset, step, None, intermediate_tensors=intermediate
                )
            last.record(stream)
            record = {
                "chunk_idx": chunk,
                "step_idx": step,
                "slot_idx": slot_idx,
                "rank": rank,
                "stage_idx": rank,
                "timestep": 0.0 if clean_pass else timestep_values[step],
                "forward_ms": 0.0,
                "stage_start_ms": 0.0,
                "stage_end_ms": 0.0,
                "comm_wait_ms": 0.0,
                "input_latent_frames": model_input.shape[2],
            }
            pending_timings.append((record, first, last))
            if kv:
                record.update(
                    pass_kind="clean_kv" if clean_pass else "denoise",
                    kv_reader_producer=reader,
                    kv_sources=[list(source) for source in kv_sources[task]],
                    kv_history_latent_frames=len(kv_sources[task]) * t_l,
                )
                release_kv(task, publishers)
            records.append(record)
            if rank != last_rank:
                forward_payload = {**prediction.tensors, "model_input": model_input}
            else:
                prediction = prediction[:, :, -t_l:]
                if clean_pass:
                    # The context pass publishes KV only: per the reference
                    # contract it never updates the sample state.
                    updated = model_input[:, :, -t_l:]
                else:
                    noise = (
                        step_noises[step][:, :, chunk * t_l : (chunk + 1) * t_l].contiguous()
                        if kv and step < last_denoise
                        else (
                            torch.randn(shape, generator=generators[chunk], device=device, dtype=torch.float32)
                            if step < last_denoise
                            else None
                        )
                    )
                    updated = update_sample(
                        prediction=prediction,
                        sample=model_input[:, :, -t_l:],
                        timestep=timestep_values[step],
                        next_timestep=timestep_values[step + 1] if step < last_denoise else None,
                        noise=noise,
                    )
                feedback_payload = {"latents": updated}
                if world == 1:
                    accept_feedback(task, feedback_payload)

        exchange_ms = 0.0
        if world > 1:
            exchange_start = time.perf_counter()
            handles = []
            postprocessors = []
            received_feedback = received_forward = None
            # The dictionary APIs send metadata synchronously. Both ranks must
            # use this SAME direction order, never issue two opposite sends first.
            if tasks[last_rank] is not None:
                if rank == last_rank:
                    feedback_bytes += _tensor_bytes({"latents": feedback_payload["latents"]})
                    with trace_scope(f"chunk_pp.metadata.send_feedback.slot{slot_idx}"):
                        handles.extend(pp.isend_tensor_dict(feedback_payload, dst=0))
                else:
                    with trace_scope(f"chunk_pp.metadata.recv_feedback.slot{slot_idx}"):
                        received_feedback, work, postprocess = pp.irecv_tensor_dict(src=last_rank)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            for src_rank in range(last_rank):
                dst = src_rank + 1
                if tasks[src_rank] is None:
                    continue
                if rank == src_rank:
                    activation_bytes += _tensor_bytes(
                        {key: value for key, value in forward_payload.items() if key != "model_input"}
                    )
                    sample_bytes += _tensor_bytes({"model_input": forward_payload["model_input"]})
                    with trace_scope(f"chunk_pp.metadata.send_forward.slot{slot_idx}"):
                        handles.extend(pp.isend_tensor_dict(forward_payload, dst=dst))
                elif rank == dst:
                    with trace_scope(f"chunk_pp.metadata.recv_forward.slot{slot_idx}"):
                        received_forward, work, postprocess = pp.irecv_tensor_dict(src=src_rank)
                    postprocessors.extend(postprocess)
                    handles.extend(work)
            # Keep all outgoing and incoming payloads alive until transfer ends.
            with trace_scope(f"chunk_pp.p2p.wait.slot{slot_idx}"):
                for handle in handles:
                    handle.wait()
            for postprocess in postprocessors:
                postprocess()
            # ``Work.wait`` establishes the dependency from the P2P work to
            # this stream. A full stream synchronization here only blocks the
            # host before it can enqueue the next ready slot.
            if received_feedback is not None:
                accept_feedback(tasks[last_rank], received_feedback)
            if received_forward is not None:
                stage_input = received_forward
            exchange_ms = (time.perf_counter() - exchange_start) * 1000
            comm_ms += exchange_ms
        if record is not None:
            record["comm_wait_ms"] = exchange_ms
        slot_records.append({"slot_idx": slot_idx, "task": task, "comm_wait_ms": exchange_ms})
    torch.accelerator.synchronize(device)
    for record, first, last in pending_timings:
        record["forward_ms"] = first.elapsed_time(last)
        record["stage_start_ms"] = origin.elapsed_time(first)
        record["stage_end_ms"] = origin.elapsed_time(last)
    barrier()
    local_wall = (time.perf_counter() - start_wall) * 1000
    payload = {
        "rank": rank,
        "stage_idx": rank,
        "layer_range": list(layer_range) if layer_range is not None else None,
        "steps": records,
        "slots": slot_records,
        "denoise_wall_ms": local_wall,
        "comm_wait_ms": comm_ms,
        "comm_wait_scope": (
            "host wall time of the per-slot tensor-dict exchange, including dict metadata, "
            "P2P submission and waits, and postprocessing; not device-communication-only time"
        ),
        "activation_bytes_sent": activation_bytes,
        "sample_bytes_sent": sample_bytes,
        "feedback_bytes_sent": feedback_bytes,
        "total_payload_bytes_sent": activation_bytes + sample_bytes + feedback_bytes,
        "denoise_peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
    }
    if kv:
        payload.update(
            kv_live_versions=kv_peak_live,
            kv_evicted_versions=kv_evicted,
            kv_retained_bytes_est=kv_peak_retained_bytes,
        )
    all_metrics = [payload]
    if world > 1:
        all_metrics = [None] * world
        dist.all_gather_object(all_metrics, payload, group=pp.cpu_group)

    gather_start = time.perf_counter()
    full_shape = (*shape[:2], t_l * chunks, *shape[3:])
    if rank == 0:
        result = torch.cat([clean_chunks[chunk] for chunk in range(chunks)], dim=2)
    else:
        result = torch.zeros(full_shape, device=device, dtype=torch.float32)
    torch.accelerator.synchronize(device)
    metrics = {
        "execution": "whole_request_layer_chunk_pipeline",
        "schedule": schedule,
        "world_size": world,
        "chunks": chunks,
        "chunk_latent_frames": t_l,
        "seed": seed,
        "latent_shape": list(shape),
        "ranks": all_metrics,
        "forward_call_unit": "layer_stage" if world > 1 else "whole_transformer",
        "stage_time_basis": "rank_local_cuda_event_origin; clocks are not aligned across ranks",
        "activation_bytes_include_model_input": False,
        "denoise_wall_ms": max(item["denoise_wall_ms"] for item in all_metrics),
        "forward_total_ms": sum(row["forward_ms"] for item in all_metrics for row in item["steps"]),
        "comm_wait_rank_sum_ms": sum(item["comm_wait_ms"] for item in all_metrics),
        "activation_bytes_sent": sum(item["activation_bytes_sent"] for item in all_metrics),
        "sample_bytes_sent": sum(item["sample_bytes_sent"] for item in all_metrics),
        "feedback_bytes_sent": sum(item["feedback_bytes_sent"] for item in all_metrics),
        "total_payload_bytes_sent": sum(item["total_payload_bytes_sent"] for item in all_metrics),
        "latent_gather_ms": (time.perf_counter() - gather_start) * 1000,
    }
    if kv:
        metrics.update(
            conditioning="per_layer_latest_kv",
            kv_history_chunks=kv_history_chunks,
            kv_source_policy=kv_source_policy,
            kv_version_policy=(
                "each chunk reads the highest finished version of its history chunks "
                "(serial + latest is the replay control of the stepwise schedule)"
                if kv_source_policy == "latest"
                else "each chunk reads only the clean pass of its history chunks (Self Forcing)"
            ),
            kv_live_versions=max(item["kv_live_versions"] for item in all_metrics),
            kv_evicted_versions=sum(item["kv_evicted_versions"] for item in all_metrics),
            kv_retained_bytes_est=max(item["kv_retained_bytes_est"] for item in all_metrics),
            kv_retained_bytes_scope=(
                "logical K/V bytes (numel * element_size) per layer stage, sampled at each "
                "release entry; summed across layers, not allocator reserved bytes"
            ),
            clean_kv_forward=True,
            clean_kv_forward_total_ms=sum(
                row["forward_ms"] for item in all_metrics for row in item["steps"] if row["pass_kind"] == "clean_kv"
            ),
            noise_contract="shared full-video initial sample and one re-noise tensor per step transition",
        )
    return result, metrics
