# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Fixed-address same-step page ring; push once to each actual consumer rank.

Each slot contains one (chunk, step, block) page. Capacity H+1 ensures every
reader of an old chunk precedes the replacing chunk. Ready/release tickets
order writers, even when different source ranks reuse the same slot.
"""

import socket

import torch
import torch.distributed as dist

from .horizontal_kv import consumers
from .kv_ipc import IPCPush
from .kv_policy import KVKey

# Failed asynchronous requests cannot safely free mapped buffers while peers
# may still access them. Keep them alive until torchrun terminates the job.
_ABORTED = []


class PushPages(IPCPush):
    def __init__(self, manager, shape, dtype):
        from cuda.bindings import driver

        self.driver = driver
        self.manager = manager
        self.device, self.group = manager.device, manager.group
        self.rank, self.world = manager.rank, len(manager.nodes)
        self.history, self.chunks = manager.policy.history_chunks, manager.chunks
        if self.chunks >= 2**31:
            raise ValueError("IPC chunk tickets require fewer than 2**31 chunks")
        self.stages, self.blocks = manager.stages, manager.blocks
        self.storage_blocks = getattr(manager, "storage_blocks", self.blocks)
        self.capacity = min(self.chunks, self.history + 1)
        self.streams = {}
        self.opened, self.peers = {}, {}
        self.hosts = ()
        self.remote_outgoing = {}
        self.remote_received = set()
        self.closed = False
        self.page_shape = tuple(shape)
        self.bytes_per_tensor = manager.page_bytes // 2
        self.buffers = self.flags = None
        error = info = None
        try:
            # No pages are needed when history is zero.
            capacity = self.capacity if self.history else 0
            self.buffers = self.allocate_buffers(shape, dtype, capacity)
            self.flags = torch.zeros(
                (2, capacity, self.stages, self.storage_blocks), dtype=torch.int32, device=self.device
            )
            info = (
                socket.gethostname(),
                str(torch.cuda.get_device_properties(self.device).uuid),
                (self.capacity, self.stages, self.storage_blocks, tuple(shape), dtype),
                self.export(self.buffers) if capacity else None,
                self.export(self.flags) if capacity else None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        entries = IPCPush.gather(self, (info, error))
        if any(e for _, e in entries):
            raise RuntimeError("IPC page allocation failed: " + str([e for _, e in entries]))
        infos = [i for i, _ in entries]
        self.hosts = tuple(i[0] for i in infos)
        if any(i[2] != info[2] for i in infos):
            raise ValueError("IPC pages require matching shapes")
        if self.manager.transport_policy.backend == "ipc":
            if len(set(self.hosts)) != 1 or len({i[1] for i in infos}) != self.world:
                raise ValueError("IPC pages require matching shapes and distinct GPUs on one host")
        elif any(
            len({infos[r][1] for r, host in enumerate(self.hosts) if host == name}) != self.hosts.count(name)
            for name in set(self.hosts)
        ):
            raise ValueError("hybrid IPC peers require distinct GPUs on each host")
        error = None
        try:
            if self.history:
                for rank, (_, _, _, data, flags) in enumerate(infos):
                    if self.hosts[rank] != self.hosts[self.rank]:
                        continue
                    self.peers[rank] = (
                        (self.buffers.data_ptr(), self.flags.data_ptr())
                        if rank == self.rank
                        else (self.open(data), self.open(flags))
                    )
                    # Per-destination streams avoid a slow reader blocking pushes
                    # to other ranks that are needed to advance that reader.
                    self.streams[rank] = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = IPCPush.gather(self, error)
        if any(errors):
            for ptr in self.opened.values():
                self.check(driver.cuIpcCloseMemHandle(ptr))
            self.opened.clear()
            raise RuntimeError("IPC page mapping failed: " + str(errors))
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group, device_ids=[self.device.index])

    def allocate_buffers(self, shape, dtype, capacity):
        return torch.empty((capacity, self.stages, self.storage_blocks, 2, *shape), dtype=dtype, device=self.device)

    def storage_index(self, key, rank=None):
        return key.block

    def index(self, key, rank=None):
        return ((key.chunk % self.capacity) * self.stages + key.step) * self.storage_blocks + self.storage_index(
            key, rank
        )

    def flag(self, base, kind, key, rank=None):
        return base + (kind * self.capacity * self.stages * self.storage_blocks + self.index(key, rank)) * 4

    def page(self, key):
        return self.buffers[key.chunk % self.capacity, key.step, self.storage_index(key)]

    def destinations(self, chunk):
        return consumers(chunk, self.chunks, self.history, self.world)

    def destinations_for(self, key):
        return self.destinations(key.chunk)

    def local_peer(self, rank):
        return self.hosts[rank] == self.hosts[self.rank]

    def source_rank(self, key):
        return key.chunk % self.world

    def global_rank(self, rank):
        return dist.get_global_rank(self.group, rank) if self.group is not None else rank

    def collect_remote(self):
        """Release completed NCCL sends while retaining unfinished CUDA buffers."""
        for identity, (works, _) in list(self.remote_outgoing.items()):
            if all(work.is_completed() for work in works):
                self.remote_outgoing.pop(identity)

    def retire_remote(self, chunk):
        """Bound source-side P2P lifetimes after every possible reader ran."""
        for (key, destination), (works, _) in list(self.remote_outgoing.items()):
            if key.chunk + self.history < chunk:
                for work in works:
                    work.wait()
                self.remote_outgoing.pop((key, destination))

    def reader_identity(self, chunk, step):
        return chunk

    def previous(self, chunk, destination, step=0, block=0):
        old = chunk - self.capacity
        while old >= 0:
            if destination in self.destinations_for(KVKey(old, step, block)):
                return old
            old -= self.capacity
        return None

    def publish(self, key, k, v):
        destinations = self.destinations_for(key)
        if not destinations:
            return
        packed = (k.contiguous(), v.contiguous())
        self.retire_remote(key.chunk)
        self.collect_remote()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        remote = []
        for dst in destinations:
            if not self.local_peer(dst):
                remote.append(dst)
                self.manager.sent_bytes[self.rank] += 2 * self.bytes_per_tensor
                continue
            stream = self.streams[dst]
            stream.wait_event(ready)
            data, flags = self.peers[dst]
            old = self.previous(key.chunk, dst, key.step, key.block)
            if old is not None:
                self.wait(stream.cuda_stream, self.flag(flags, 1, key, dst), old + 1)
            address = data + self.index(key, dst) * 2 * self.bytes_per_tensor
            for i, tensor in enumerate(packed):
                self.copy(
                    address + i * self.bytes_per_tensor, tensor.data_ptr(), self.bytes_per_tensor, stream.cuda_stream
                )
                tensor.record_stream(stream)
            self.write(stream.cuda_stream, self.flag(flags, 0, key, dst), key.chunk + 1)
            if dst != self.rank:
                self.manager.sent_bytes[self.rank] += 2 * self.bytes_per_tensor
        if remote:
            operations = []
            for dst in remote:
                operations.extend(
                    dist.P2POp(dist.isend, tensor, self.global_rank(dst), self.group) for tensor in packed
                )
            works = dist.batch_isend_irecv(operations)
            for index, dst in enumerate(remote):
                self.remote_outgoing[key, dst] = (tuple(works[index * 2 : index * 2 + 2]), packed)

    def read(self, key):
        return self.read_many((key,))[key]

    def read_many(self, keys):
        pages, operations, pending = {}, [], []
        for key in keys:
            source = self.source_rank(key)
            if not self.local_peer(source):
                data = self.page(key)
                if key not in self.remote_received:
                    pending.append(key)
                    operations.extend(
                        dist.P2POp(dist.irecv, tensor, self.global_rank(source), self.group) for tensor in data
                    )
                pages[key] = (data[0], data[1], None)
                continue
            self.wait(
                torch.cuda.current_stream(self.device).cuda_stream,
                self.flag(self.flags.data_ptr(), 0, key),
                key.chunk + 1,
            )
            data = self.page(key)
            pages[key] = (data[0], data[1], None)
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
            self.remote_received.update(pending)
        return pages

    def release(self, key, consumer, step=None):
        if not self.local_peer(self.source_rank(key)):
            if self.destinations_for(key)[self.rank] != self.reader_identity(consumer, step):
                return
            self.remote_received.discard(key)
            if self.source_rank(key) != self.rank:
                self.manager.received_bytes[self.rank] += 2 * self.bytes_per_tensor
            return
        if self.destinations_for(key)[self.rank] != self.reader_identity(consumer, step):
            return
        # Current stream has finished copying pages into the separate cat output.
        self.write(
            torch.cuda.current_stream(self.device).cuda_stream, self.flag(self.flags.data_ptr(), 1, key), key.chunk + 1
        )
        if self.source_rank(key) != self.rank:
            self.manager.received_bytes[self.rank] += 2 * self.bytes_per_tensor

    def close(self, abort=False):
        if self.closed:
            return
        if abort:
            _ABORTED.append(self)
            self.closed = True
            return
        for works, _ in self.remote_outgoing.values():
            for work in works:
                work.wait()
        self.remote_outgoing.clear()
        self.remote_received.clear()
        for stream in self.streams.values():
            done = torch.cuda.Event()
            done.record(stream)
            torch.cuda.current_stream(self.device).wait_event(done)
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group, device_ids=[self.device.index])
        for ptr in self.opened.values():
            self.check(self.driver.cuIpcCloseMemHandle(ptr))
        self.opened.clear()
        self.peers.clear()
        dist.barrier(group=self.group, device_ids=[self.device.index])
        self.buffers = self.flags = None
        self.closed = True
