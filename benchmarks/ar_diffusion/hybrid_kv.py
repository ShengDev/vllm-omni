# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-block IPC/NCCL publication with the native vLLM Wan attention.

Only the KV transport is borrowed from the archived WaveServe tick plan.
The native chunk plan, native model, sampler and paged FA3 remain in use.
Layer-major rings keep each attention pool contiguous in token order.
"""

import ctypes
import sys
from contextlib import contextmanager

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVState, NoisyLayerContext

from .legacy_tick.cell_schedule import Vertical
from .legacy_tick.kv_policy import KVKey, KVPolicy
from .legacy_tick.kv_storage import KVStorage
from .legacy_tick.kv_transport import KVTransport
from .legacy_tick.tick_latest import StaticTickLatest, TickPages, TickPlan


class LayerMajorPages(TickPages):
    def allocate_buffers(self, shape, dtype, capacity):
        return torch.empty((self.storage_blocks, 2, capacity, self.stages, *shape), dtype=dtype, device=self.device)

    def export(self, tensor):
        pointer = tensor.data_ptr()
        handle = self.check(self.driver.cuIpcGetMemHandle(pointer))
        base, _ = self.check(self.driver.cuMemGetAddressRange(pointer))
        raw = bytes(handle.reserved) if hasattr(handle, "reserved") else ctypes.string_at(handle.getPtr(), 64)
        return raw, pointer - int(base)

    def open(self, descriptor):
        raw, offset = descriptor
        if raw not in self.opened:
            handle = self.driver.CUipcMemHandle()
            if hasattr(handle, "reserved"):
                handle.reserved = raw
            else:
                if len(raw) != 64:
                    raise ValueError("invalid CUDA IPC handle")
                ctypes.memmove(handle.getPtr(), raw, 64)
            flag = int(self.driver.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS.value)
            self.opened[raw] = int(self.check(self.driver.cuIpcOpenMemHandle(handle, flag)))
        return self.opened[raw] + offset

    def page(self, key):
        block = self.storage_index(key)
        return tuple(self.buffers[block, field, key.chunk % self.capacity, key.step] for field in range(2))

    def address(self, base, key, destination, field):
        block = self.storage_index(key, destination)
        slot = key.chunk % self.capacity * self.stages + key.step
        return base + ((block * 2 + field) * self.capacity * self.stages + slot) * self.bytes_per_tensor

    def publish(self, key, k, v):
        packed = (k.contiguous(), v.contiguous())
        if not self.local_only:
            self.staged[key] = packed
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        for destination in self.destinations_for(key):
            stream = self.streams[destination]
            stream.wait_event(ready)
            data, flags = self.peers[destination]
            old = self.previous(key.chunk, destination, key.step, key.block)
            if old is not None:
                self.wait(stream.cuda_stream, self.flag(flags, 1, key, destination), old + 1)
            for field, value in enumerate(packed):
                self.copy(
                    self.address(data, key, destination, field),
                    value.data_ptr(),
                    self.bytes_per_tensor,
                    stream.cuda_stream,
                )
                value.record_stream(stream)
            self.write(stream.cuda_stream, self.flag(flags, 0, key, destination), key.chunk + 1)
            if destination != self.rank:
                self.manager.sent_bytes[self.rank] += 2 * self.bytes_per_tensor
        if not self.local_only and self.overlap:
            self.commit_round()


class NativeTickManager(StaticTickLatest):
    page_type = LayerMajorPages


def validate_native_plan(plan, schedule, history=6):
    tick = TickPlan(schedule.tasks, schedule.stages, history)
    for task in schedule.tasks:
        native = plan.sources((task.chunk, task.step), task.rank)
        wanted = tuple(KVKey(x.version[0], x.version[1], task.block) for x in native)
        if tick.reads[task] != wanted:
            raise ValueError(f"native/tick Latest KV labels differ: {task}")
        if plan.completion_slot((task.chunk, task.step), task.rank % plan.schedule.layer_groups) != task.tick // (
            30 // plan.schedule.layer_groups
        ):
            raise ValueError(f"native/tick execution slot differs: {task}")
        # Native idle ranks have no forward hook to post a receive. Verify that
        # every producer-round recipient is active in that same slot.
        for dst in tick.destinations.get(KVKey(*task.key), {}):
            if plan.task(task.tick // (30 // plan.schedule.layer_groups), dst) is None:
                raise ValueError(f"idle native rank must receive tick payload: {task}, {dst}")
    return tick


class HybridNoisyKVState(NoisyKVState):
    def __init__(self, cache, model, group):
        super().__init__(cache)
        self.model, self.group = model, group
        self.manager = None
        self.prepared = {}
        self.current = None

    def begin_request(self, req, plan, *, chunk_tokens, t0=0):
        if self.manager is not None or t0 != 0:
            raise ValueError("hybrid benchmark supports one request at a time")
        super().begin_request(req, plan, chunk_tokens=chunk_tokens, t0=t0)
        groups, stages = plan.schedule.layer_groups, plan.schedule.stages
        self.schedule = Vertical(30 // groups).schedule(30, stages, plan.schedule.chunks)
        validate_native_plan(plan, self.schedule)
        spec = self.cache.spec
        self.manager = NativeTickManager(
            KVPolicy(selection="latest", history_chunks=spec.max_history_chunks, clean_store=True),
            KVStorage(),
            tuple(range(plan.world)),
            stages,
            device=self.cache.pool.k_pools[0].device,
            group=self.group,
            distributed=True,
            transport=KVTransport("hybrid", round_overlap=True),
            tasks=self.schedule.tasks,
            shape=(1, spec.num_kv_heads, chunk_tokens, spec.head_size),
            blocks=30,
            chunks=plan.schedule.chunks,
            dtype=self.cache.pool.k_pools[0].dtype,
            token_major=True,
        )
        push = self.manager.push
        device = push.device
        for chunk in range(plan.schedule.chunks):
            step = self._rank // groups
            task = self.schedule.task(chunk, step, self.model.start_layer)
            keys = self.manager.plan.reads[task]
            ids = [k.chunk % push.capacity * stages + k.step for k in keys]
            write = chunk % push.capacity * stages + step
            ids.append(write)
            width = spec.max_history_chunks + 1

            def gpu(values, dtype):
                return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)

            table = gpu([ids + [0] * (width - len(ids))], torch.int32)
            seq_len = len(ids) * chunk_tokens
            seq_lens = gpu([seq_len], torch.int32)
            query_locs = gpu([0, chunk_tokens], torch.int32)
            slots = torch.arange(chunk_tokens, device=device, dtype=torch.long) + write * chunk_tokens
            contexts = []
            for local in range(self.model.local_num_layers):
                pools = [push.buffers[local, field].view(-1, spec.num_kv_heads, spec.head_size) for field in range(2)]
                assert all(value.is_contiguous() for value in pools)
                contexts.append(
                    NoisyLayerContext(
                        local,
                        pools[0],
                        pools[1],
                        chunk_tokens,
                        slots,
                        table,
                        query_locs,
                        seq_lens,
                        chunk_tokens,
                        width * chunk_tokens,
                        seq_len,
                    )
                )
            self.prepared[chunk] = contexts

    def prepare(self, tasks):
        if len(tasks) != 1:
            raise ValueError("hybrid benchmark requires one chunk per slot")
        _, self.current = tasks[0]
        return [self.prepared[self.current[0]]]

    def attend(self, original, inputs, query, key, value, *args, **kwargs):
        chunk, step = self.current
        block = self.model.start_layer + int(inputs.layer_idx)
        task = self.schedule.task(chunk, step, block)
        self.manager.prepare(self.schedule.by_tick[task.tick])
        self.manager.publish(task, key.unsqueeze(0), value.unsqueeze(0))
        push = self.manager.push
        reads = self.manager.plan.reads[task]
        own = KVKey(chunk, step, block)
        waits = reads + ((own,) if self._rank in push.all_destinations(own) else ())
        push.read_many(waits)
        output = original(inputs, query, key, value, *args, **kwargs)
        for previous in reads:
            # Releases are on the attention stream, after FA3 has read the page.
            push.release(previous, chunk, step)
        self.manager.commit()
        return output

    def publish(self, tasks):
        pass

    def exchange(self, slot):
        if self.manager is not None:
            self.bytes_sent = self.manager.sent_bytes[self._rank]
            self.bytes_received = self.manager.received_bytes[self._rank]
        return []

    def evict(self, slot):
        pass

    def await_ready(self, slot):
        return 0

    def drain(self):
        pass

    def end_request(self, req):
        if self.manager is not None:
            self.bytes_sent = self.manager.sent_bytes[self._rank]
            self.bytes_received = self.manager.received_bytes[self._rank]
            if sys.exc_info()[0] is not None:
                self.manager.push.close(abort=True)
            else:
                self.manager.close()
            self.manager = None
        self.prepared.clear()
        super().end_request(req)

    def reset_all(self):
        if self.manager is not None:
            self.manager.push.close(abort=True)
            self.manager = None
        self.prepared.clear()
        super().reset_all()


@contextmanager
def native_hybrid_attention(state):
    from vllm_omni.diffusion.models.waveserve_wan import transformer

    original = transformer.paged_write_attn
    transformer.paged_write_attn = lambda *a, **k: state.attend(original, *a, **k)
    try:
        yield
    finally:
        transformer.paged_write_attn = original
        if state.manager is not None:
            state.manager.push.close(abort=True)
