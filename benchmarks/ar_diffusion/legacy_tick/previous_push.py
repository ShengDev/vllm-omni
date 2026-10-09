# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""SSL per-layer IPC push; physical arrival never determines KV visibility.

Denoise versions have one query per destination. Clean versions live through
that destination's last history query. Two slots per denoise producer and H+1
clean slots retain those lifetimes without allocating H copies of every version.
"""

import math
import socket

import torch
import torch.distributed as dist

from .kv_ipc import IPCPush
from .previous_kv import PreviousTickCache

_ABORTED = []


def readers(chunk, source, destination, chunks, history, stages):
    """Inclusive query-chunk interval; None when this destination never reads it."""
    if destination > source or not 0 <= chunk < chunks:
        return None
    first = chunk + source + 1 - destination
    last = min(chunks - 1, chunk + history)
    if source != stages - 1:
        last = min(last, first)
    return (first, last) if first <= last else None


def labels(chunk, step, block, history, stages):
    return [(c, min(stages - 1, chunk + step - 1 - c), block) for c in range(max(0, chunk - history), chunk)]


class PagePlan:
    def __init__(self, chunks, history, stages, blocks):
        self.chunks, self.history, self.stages, self.blocks = chunks, history, stages, blocks
        self.counts = []
        self.offsets = []
        for destination in range(stages):
            counts, offsets, offset = [], [], 0
            for source in range(stages):
                count = (
                    min(chunks, history + 1 if source == stages - 1 else 2)
                    if source >= destination and source + 1 - destination <= history
                    else 0
                )
                counts.append(count)
                offsets.append(offset)
                offset += count * blocks
            self.counts.append(counts)
            self.offsets.append(offsets)

    def size(self, destination):
        return sum(self.counts[destination]) * self.blocks

    def index(self, destination, chunk, source, block):
        count = self.counts[destination][source]
        if not count:
            raise ValueError("page has no consumer slots")
        return self.offsets[destination][source] + (chunk % count) * self.blocks + block


class PreviousPush(IPCPush):
    """Fixed receive slots with one writer per source; GPU ready/release tickets."""

    def __init__(self, *, chunks, history, stages, blocks, shape, dtype, device, group=None, token_major=False):
        from cuda.bindings import driver

        self.driver = driver
        self.device, self.group = torch.device(device), group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if self.world != stages or chunks >= 2**31:
            raise ValueError("SSL IPC requires one rank per stage and fewer than 2**31 chunks")
        self.plan = PagePlan(chunks, history, stages, blocks)
        self.chunks, self.history, self.stages = chunks, history, stages
        self.token_major = token_major
        self.shape = tuple(shape)
        self.page_shape = (shape[0], shape[2], shape[1], shape[3]) if token_major else tuple(shape)
        self.tensor_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        self.streams, self.opened, self.peers = {}, {}, {}
        self.sent = self.received = 0
        self.closed = False
        self.buffers = self.flags = None
        error = info = None
        try:
            size = self.plan.size(self.rank)
            self.buffers = torch.empty((size, 2, *self.page_shape), device=self.device, dtype=dtype)
            self.flags = torch.zeros((2, size), device=self.device, dtype=torch.int32)
            contract = (chunks, history, stages, blocks, tuple(shape), dtype, token_major)
            info = (
                socket.gethostname(),
                str(torch.cuda.get_device_properties(self.device).uuid),
                contract,
                self.export(self.buffers) if size else None,
                self.export(self.flags) if size else None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        entries = self.gather((info, error))
        if any(e for _, e in entries):
            raise RuntimeError("SSL IPC allocation failed: " + str([e for _, e in entries]))
        infos = [i for i, _ in entries]
        if (
            len({i[0] for i in infos}) != 1
            or len({i[1] for i in infos}) != stages
            or any(i[2] != info[2] for i in infos)
        ):
            raise ValueError("SSL IPC requires matching contracts on distinct GPUs of one host")
        error = None
        try:
            for rank, (_, _, _, data, flags) in enumerate(infos):
                if data is not None:
                    self.peers[rank] = (
                        (self.buffers.data_ptr(), self.flags.data_ptr())
                        if rank == self.rank
                        else (self.open(data), self.open(flags))
                    )
                    self.streams[rank] = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self.gather(error)
        if any(errors):
            for pointer in self.opened.values():
                self.check(driver.cuIpcCloseMemHandle(pointer))
            self.opened.clear()
            raise RuntimeError("SSL IPC mapping failed: " + str(errors))
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=group, device_ids=[self.device.index])

    def flag(self, base, destination, kind, index):
        return base + (kind * self.plan.size(destination) + index) * 4

    def publish(self, chunk, source, block, key, value):
        destinations = [
            d for d in range(source + 1) if readers(chunk, source, d, self.chunks, self.history, self.stages)
        ]
        if not destinations:
            return

        def convert(x):
            return x.transpose(1, 2) if self.token_major else x

        packed = tuple(convert(x).contiguous() for x in (key, value))
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        for destination in destinations:
            stream = self.streams[destination]
            stream.wait_event(ready)
            data, flags = self.peers[destination]
            index = self.plan.index(destination, chunk, source, block)
            old = chunk - self.plan.counts[destination][source]
            if old >= 0:
                self.wait(stream.cuda_stream, self.flag(flags, destination, 1, index), old + 1)
            target = data + index * 2 * self.tensor_bytes
            for i, tensor in enumerate(packed):
                self.copy(target + i * self.tensor_bytes, tensor.data_ptr(), self.tensor_bytes, stream.cuda_stream)
                tensor.record_stream(stream)
            self.write(stream.cuda_stream, self.flag(flags, destination, 0, index), chunk + 1)
            if destination != self.rank:
                self.sent += 2 * self.tensor_bytes

    def read_page(self, chunk, source, block):
        index = self.plan.index(self.rank, chunk, source, block)
        self.wait(
            torch.cuda.current_stream(self.device).cuda_stream,
            self.flag(self.flags.data_ptr(), self.rank, 0, index),
            chunk + 1,
        )
        page = self.buffers[index]
        return tuple(x.transpose(1, 2) for x in page) if self.token_major else tuple(page)

    def release_page(self, chunk, source, block, query):
        if query != readers(chunk, source, self.rank, self.chunks, self.history, self.stages)[1]:
            return
        index = self.plan.index(self.rank, chunk, source, block)
        self.write(
            torch.cuda.current_stream(self.device).cuda_stream,
            self.flag(self.flags.data_ptr(), self.rank, 1, index),
            chunk + 1,
        )
        if source != self.rank:
            self.received += 2 * self.tensor_bytes

    def close(self, abort=False):
        if self.closed:
            return
        if abort:
            _ABORTED.append(self)
            self.closed = True
            return
        for stream in self.streams.values():
            done = torch.cuda.Event()
            done.record(stream)
            torch.cuda.current_stream(self.device).wait_event(done)
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group, device_ids=[self.device.index])
        for pointer in self.opened.values():
            self.check(self.driver.cuIpcCloseMemHandle(pointer))
        self.opened.clear()
        self.peers.clear()
        dist.barrier(group=self.group, device_ids=[self.device.index])
        self.buffers = self.flags = None
        self.closed = True


class PushCache(PreviousTickCache):
    def __init__(self, push):
        super().__init__(push.history)
        self.push = push

    def stage(self, block, key, value):
        # publish() records the source tensors on every destination stream.
        # No persistent pending clone or model-round commit is necessary.
        pass

    def context(self, step, block, chunk, key, value, **kwargs):
        wanted = labels(chunk, step, block, self.history, self.push.stages)
        self.pages[block] = {c: (s, *self.push.read_page(c, s, b)) for c, s, b in wanted}
        result = super().context(step, block, chunk, key, value, **kwargs)
        # Both cat and mirrored ring own separate copies from the receive slots.
        for c, s, b in wanted:
            self.push.release_page(c, s, b, chunk)
        self.push.publish(chunk, step, block, key, value)
        return result
