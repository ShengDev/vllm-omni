# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Hybrid pages with one deterministic, producer-tick NCCL round.

IPC publishes from attention. By default remote pages exchange at tick end.
Opt-in overlap launches the same round at Q/K/V publication on a dedicated
stream; consumers wait for the chosen page's event before copying history.
"""

import torch
import torch.distributed as dist

from .horizontal_kv_ipc import PushPages
from .kv_policy import KVKey


class RoundPushPages(PushPages):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.staged = {}
        self.round_tasks = ()
        self.overlap = self.manager.transport_policy.round_overlap
        self.round_stream = torch.cuda.Stream(device=self.device) if self.overlap else None
        self.remote_ready = {}
        self.round_submitted = False

    def prepare_round(self, tasks):
        self.round_tasks = tuple(sorted(tasks, key=lambda t: (t.chunk, t.step, t.block, t.rank)))
        self.round_submitted = False
        if self.overlap and sum(t.rank == self.rank for t in self.round_tasks) > 1:
            raise ValueError("overlapped rounds require at most one task per rank and tick")

    def publish(self, key, k, v):
        # Parent only handles local IPC edges; no dynamic NCCL calls remain.
        self.staged[key] = (k.contiguous(), v.contiguous())
        super().publish(key, k, v)
        if self.overlap:
            # Q/K/V are ready before attention and FFN. The static round may
            # launch here even if other ranks reach it later (or are idle).
            self.commit_round()

    def destinations_for(self, key):
        return {r: c for r, c in self.all_destinations(key).items() if self.local_peer(r)}

    def all_destinations(self, key):
        return super().destinations_for(key)

    def commit_round(self):
        if self.round_submitted:
            return
        self.round_submitted = True
        operations = []
        received = []
        for task in self.round_tasks:
            key = KVKey(task.chunk, task.step, task.block)
            for destination in sorted(self.all_destinations(key)):
                if self.hosts[task.rank] == self.hosts[destination]:
                    continue
                if self.rank == task.rank:
                    tensors = self.staged[key]
                    op, peer = dist.isend, destination
                    self.manager.sent_bytes[self.rank] += 2 * self.bytes_per_tensor
                elif self.rank == destination:
                    tensors = self.page(key)
                    op, peer = dist.irecv, task.rank
                    self.remote_received.add(key)
                    received.append(key)
                    self.manager.received_bytes[self.rank] += 2 * self.bytes_per_tensor
                else:
                    continue
                operations.extend(dist.P2POp(op, tensor, self.global_rank(peer), self.group) for tensor in tensors)
        if operations:
            if self.overlap:
                # The event precedes this tick's history waits. In particular,
                # old slot readers from earlier ticks have finished their cat.
                # Never wait on the compute stream again after launching here:
                # that would create a cycle with this tick's history read.
                self.round_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.round_stream):
                    self.exchange(operations)
                    done = torch.cuda.Event()
                    done.record(self.round_stream)
                    # Retain source storage until NCCL is done, without an
                    # unbounded Python queue or a compute-stream fence.
                    for tensors in self.staged.values():
                        for tensor in tensors:
                            tensor.record_stream(self.round_stream)
                for key in received:
                    self.remote_ready[key] = done
            else:
                self.exchange(operations)
        # Both paths establish NCCL completion on their chosen stream before
        # dropping producer references; overlap also records allocator use.
        self.staged.clear()

    def exchange(self, operations):
        for work in dist.batch_isend_irecv(operations):
            work.wait()

    def read_many(self, keys):
        local = [key for key in keys if self.local_peer(self.source_rank(key))]
        pages = super().read_many(local)
        for key in keys:
            if key in pages:
                continue
            if key not in self.remote_received:
                raise RuntimeError(f"remote page missing from completed producer round: {key}")
            if self.overlap:
                torch.cuda.current_stream(self.device).wait_event(self.remote_ready[key])
            data = self.page(key)
            pages[key] = (data[0], data[1], None)
        return pages

    def release(self, key, consumer, step=None):
        if self.local_peer(self.source_rank(key)):
            return super().release(key, consumer, step)
        if self.all_destinations(key)[self.rank] == self.reader_identity(consumer, step):
            self.remote_received.remove(key)
            self.remote_ready.pop(key, None)
            # A later LOCAL IPC producer may reuse this same slot. Its release
            # wait does not know which transport supplied the previous page.
            self.write(
                torch.cuda.current_stream(self.device).cuda_stream,
                self.flag(self.flags.data_ptr(), 1, key),
                key.chunk + 1,
            )

    def close(self, abort=False):
        if not abort and not self.closed and self.round_stream is not None:
            self.round_stream.synchronize()
            self.remote_ready.clear()
        super().close(abort=abort)
