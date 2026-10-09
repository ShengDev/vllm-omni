# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Version directory, ownership, bounded replicas and tensor transport.

All ranks enter prepare/commit in the same order. Only metadata is gathered;
KV payloads are point-to-point, never all-gathered or globally broadcast.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .kv_policy import KVKey


@dataclass(frozen=True)
class Page:
    key: KVKey
    producer: int
    owner: int
    signature: tuple
    nbytes: int
    clean: bool


class KVManager:
    def __init__(self, policy, storage, nodes, stages, device="cpu", group=None, distributed=False, transport=None):
        from .kv_transport import KVTransport

        self.transport_policy = KVTransport() if transport is None else transport
        if not isinstance(self.transport_policy, KVTransport):
            raise TypeError("transport must be a KVTransport configuration")
        if self.transport_policy.backend in ("ipc", "hybrid") and (
            not distributed or torch.device(device).type != "cuda"
        ):
            raise ValueError("IPC/hybrid KV transport requires distributed CUDA execution")
        self.transport = None
        self.policy, self.storage, self.nodes = policy, storage, tuple(nodes)
        if not nodes or (storage.clean_ranks is not None and max(storage.clean_ranks) >= len(nodes)):
            raise ValueError("clean_ranks must belong to the execution group")
        self.stages, self.device, self.group = stages, torch.device(device), group
        self.distributed = distributed
        self.rank = dist.get_rank(group) if distributed else 0
        self.directory, self.owned, self.pending = {}, {}, {}
        self.cache, self.cache_bytes = OrderedDict(), [0] * len(nodes)
        self.transient, self.reads = {}, {}
        self.sent_bytes, self.received_bytes = [0] * len(nodes), [0] * len(nodes)

    def local(self, rank):
        return not self.distributed or rank == self.rank

    def gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * len(self.nodes)
        dist.all_gather_object(values, value, group=self.group)
        return values

    def transfer(self, transfers, *, kv=False):
        """Globally ordered (identity, source, destination, signature, source tensors)."""
        if kv and self.transport_policy.backend == "ipc":
            if self.transport is None:
                from .kv_ipc import IPCPush

                self.transport = IPCPush(self.transport_policy, self.device, self.group)
            return self.transport.transfer(transfers)
        result, operations, buffers = {}, [], []
        for identity, src, dst, signature, values in transfers:
            if not self.distributed or src == dst:
                if self.local(dst):
                    result[identity] = values
                continue
            if self.rank == dst:
                tensors = tuple(
                    None if shape is None else torch.empty(shape, dtype=dtype, device=self.device)
                    for shape, dtype in signature
                )
                result[identity] = tensors
                for tensor in tensors:
                    if tensor is not None:
                        operations.append(
                            dist.P2POp(
                                dist.irecv,
                                tensor,
                                dist.get_global_rank(self.group, src) if self.group is not None else src,
                                self.group,
                            )
                        )
            if self.rank == src:
                for tensor in values:
                    if tensor is not None:
                        buffer = tensor.contiguous()
                        buffers.append(buffer)
                        operations.append(
                            dist.P2POp(
                                dist.isend,
                                buffer,
                                dist.get_global_rank(self.group, dst) if self.group is not None else dst,
                                self.group,
                            )
                        )
        # Groups are initialized by metadata collectives before subset P2P calls.
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
        return result

    def publish(self, task, key, value, raw=None):
        identity = KVKey(task.chunk, task.step, task.block)
        if identity in self.pending or identity in self.directory:
            raise ValueError(f"duplicate KV publication: {identity}")
        if key.ndim != 4 or value.shape != key.shape or value.dtype != key.dtype:
            raise ValueError("KV must have matching B,H,tokens,D shape and dtype")
        if self.policy.selection == "clean" and task.step != self.stages - 1:
            # Current noisy K/V is consumed directly by attention. No future
            # chunk may read it, so only publish the final clean refresh.
            return
        values = (key, value, raw if self.policy.rebase_sink and task.chunk == 0 else None)
        tensors = tuple(None if x is None else x.detach().clone() for x in values)
        signature = tuple((None, None) if x is None else (tuple(x.shape), x.dtype) for x in tensors)
        size = sum(x.numel() * x.element_size() for x in tensors if x is not None)
        clean = self.policy.clean_store and task.step == self.stages - 1
        self.pending[identity] = (Page(identity, task.rank, task.rank, signature, size, clean), tensors)

    def commit(self):
        pages = sorted(
            (p for batch in self.gather([p for p, _ in self.pending.values()]) for p in batch), key=lambda p: p.key
        )
        if len({p.key for p in pages}) != len(pages):
            raise ValueError("duplicate distributed publication")
        used = [0] * len(self.nodes)
        for p in self.directory.values():
            if p.clean:
                used[p.owner] += p.nbytes
        assigned, transfers = [], []
        # Placement is computed identically everywhere, including capacity failure.
        for p in pages:
            owner = self.storage.owner(p.producer, p.nbytes, self.nodes, used) if p.clean else p.producer
            if p.clean:
                used[owner] += p.nbytes
            assigned.append(Page(p.key, p.producer, owner, p.signature, p.nbytes, p.clean))
            tensors = self.pending[p.key][1] if self.local(p.producer) else None
            transfers.append((p.key, p.producer, owner, p.signature, tensors))
        received = self.transfer(transfers, kv=True)
        for p in assigned:
            self.directory[p.key] = p
            if self.local(p.owner):
                self.owned[p.key] = received[p.key]
            if p.owner != p.producer:
                self.sent_bytes[p.producer] += p.nbytes
                self.received_bytes[p.owner] += p.nbytes
        self.pending.clear()  # Drop producer copies after owner transfer completes.
        if self.policy.selection == "latest":
            newest = {}
            for k in self.directory:
                newest[k.chunk, k.block] = max(k.step, newest.get((k.chunk, k.block), -1))
            self.discard([k for k in self.directory if k.step < newest[k.chunk, k.block]])

    def prepare(self, tasks):
        self.transient, self.reads = {}, {}
        requested = []
        for task in tasks:
            keys = self.policy.select(task, self.directory)
            self.reads[task] = keys
            if not self.local(task.rank):
                continue
            for key in keys:
                page, identity = self.directory[key], (task.rank, key)
                if page.owner == task.rank:
                    self.transient[identity] = self.owned[key]
                elif identity in self.cache:
                    self.transient[identity] = self.cache[identity]
                    self.cache.move_to_end(identity)
                else:
                    requested.append(identity)
        requests = sorted(set(x for batch in self.gather(requested) for x in batch))
        transfers = []
        for dst, key in requests:
            p = self.directory[key]
            transfers.append(
                ((dst, key), p.owner, dst, p.signature, self.owned.get(key) if self.local(p.owner) else None)
            )
        received = self.transfer(transfers, kv=True)
        self.transient.update(received)
        for identity, tensors in received.items():
            dst, key = identity
            size = self.directory[key].nbytes
            if size <= self.storage.replica_cache_bytes:
                while self.cache_bytes[dst] + size > self.storage.replica_cache_bytes:
                    old = next(k for k in self.cache if k[0] == dst)
                    self.cache.pop(old)
                    self.cache_bytes[dst] -= self.directory[old[1]].nbytes
                self.cache[identity] = tensors
                self.cache_bytes[dst] += size
        for dst, key in requests:
            p = self.directory[key]
            self.sent_bytes[p.owner] += p.nbytes
            self.received_bytes[dst] += p.nbytes

    def context_for(self, task):
        manager = self

        class Context:
            def context(self, step, block, chunk, key, value, *, unrotated_key=None, rotate=None):
                if (chunk, step, block) != (task.chunk, task.step, task.block):
                    raise ValueError("model KV request does not match scheduled task")
                if manager.policy.rebase_sink and (unrotated_key is None or rotate is None):
                    raise ValueError("sink rebasing requires raw keys and rotation")
                keys = manager.reads[task]
                parts = []
                recent = sum(k.chunk >= manager.policy.sink_chunks for k in keys)
                for k in keys:
                    previous, val, raw = manager.transient[task.rank, k]
                    if manager.policy.rebase_sink and k.chunk == 0:
                        previous = rotate(raw, chunk - recent - 1)
                    parts.append((previous, val))
                parts.append((key, value))
                manager.publish(task, key, value, unrotated_key)
                return tuple(torch.cat([p[i] for p in parts], dim=2) for i in (0, 1))

        return Context()

    def discard(self, keys):
        for key in keys:
            page = self.directory[key]
            self.owned.pop(key, None)
            for identity in [i for i in self.cache if i[1] == key]:
                self.cache.pop(identity)
                self.cache_bytes[identity[0]] -= page.nbytes
            del self.directory[key]

    def finish(self, next_chunk):
        self.transient.clear()  # Readers have finished; replicas can now be evicted.
        self.reads.clear()
        if self.policy.history_chunks is not None:
            oldest = max(0, next_chunk - self.policy.history_chunks)
            self.discard([k for k in self.directory if self.policy.sink_chunks <= k.chunk < oldest])

    def close(self):
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        self.directory.clear()
        self.owned.clear()
        self.pending.clear()
        self.cache.clear()
        self.transient.clear()
        self.reads.clear()
        self.cache_bytes = [0] * len(self.nodes)
