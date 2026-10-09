# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Static Same Step dependencies; optional bounded producer-push IPC pages."""

import sys

import torch

from .kv_manager import KVManager
from .kv_policy import KVKey


def consumers(chunk, chunks, history, world):
    """Last reader on each destination; one publication can serve repeated reads."""
    return {c % world: c for c in range(chunk + 1, min(chunks, chunk + history + 1))}


class StaticSameStep(KVManager):
    def __init__(self, *args, shape, blocks, chunks, dtype, token_major=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_major = token_major
        self.shape = (shape[0], shape[2], shape[1], shape[3]) if token_major else tuple(shape)
        self.blocks, self.chunks, self.dtype = blocks, chunks, dtype
        self.signature = ((self.shape, dtype), (self.shape, dtype), (None, None))
        self.page_bytes = 2 * torch.empty((), dtype=dtype).element_size()
        for n in shape:
            self.page_bytes *= n
        self.push = None
        if self.transport_policy.backend in ("ipc", "hybrid"):
            self.push = self.make_push()

    def make_push(self):
        if self.transport_policy.backend == "hybrid":
            from .horizontal_kv_round import RoundPushPages

            return RoundPushPages(self, self.shape, self.dtype)
        from .horizontal_kv_ipc import PushPages

        return PushPages(self, self.shape, self.dtype)

    def keys_for(self, task):
        return tuple(KVKey(c, task.step, task.block) for c in self.policy.history(task.chunk))

    def gather(self, value):
        # Static execution has no healthy-path status collective. An exception
        # propagates to torchrun, which terminates peers; restart the failed job.
        return [value]

    def prepare(self, tasks):
        if self.push is not None and hasattr(self.push, "prepare_round"):
            self.push.prepare_round(tasks)
        self.transient, self.reads = {}, {}
        transfers = []
        for task in tasks:
            keys = self.keys_for(task)
            self.reads[task] = keys
            if self.push is not None:
                continue
            for key in keys:
                owner = key.chunk % len(self.nodes)
                transfers.append(
                    (
                        (task.rank, key),
                        owner,
                        task.rank,
                        self.signature,
                        self.owned.get(key) if self.local(owner) else None,
                    )
                )
                if owner != task.rank:
                    self.sent_bytes[owner] += self.page_bytes
                    self.received_bytes[task.rank] += self.page_bytes
        if self.push is None:
            self.transient = self.transfer(transfers, kv=True)

    def publish(self, task, key, value, raw=None):
        if (
            tuple(key.shape) != self.shape
            or value.shape != key.shape
            or key.dtype != self.dtype
            or value.dtype != self.dtype
        ):
            raise ValueError("static KV shape/dtype changed")
        identity = KVKey(task.chunk, task.step, task.block)
        if self.push is not None:
            self.push.publish(identity, key, value)
        else:
            self.pending[identity] = (key.detach().clone(), value.detach().clone(), None)

    def commit(self):
        if self.push is not None and hasattr(self.push, "commit_round"):
            self.push.commit_round()
        if self.push is None:
            self.owned.update(self.pending)
            self.pending.clear()

    def context_for(self, task):
        manager = self

        class Context:
            def context(self, step, block, chunk, key, value, *, unrotated_key=None, rotate=None, token_major=False):
                if (chunk, step, block) != (task.chunk, task.step, task.block):
                    raise ValueError("KV request differs from static task")
                if token_major != manager.token_major:
                    raise ValueError("static KV layout differs from model attention")
                if token_major:
                    key, value = key.transpose(1, 2), value.transpose(1, 2)
                # Publish before history waits: Q/K/V depend on block input,
                # so this safely overlaps push with the producer attention/FFN.
                manager.publish(task, key, value)
                parts = []
                pages = manager.push.read_many(manager.reads[task]) if manager.push is not None else None
                for k in manager.reads[task]:
                    page = pages[k] if pages is not None else manager.transient[task.rank, k]
                    parts.append(page[:2])
                parts.append((key, value))
                result = tuple(torch.cat([p[i] for p in parts], dim=1 if token_major else 2) for i in (0, 1))
                if token_major:
                    result = tuple(x.transpose(1, 2) for x in result)
                if manager.push is not None:
                    for k in manager.reads[task]:
                        manager.push.release(k, chunk, step)
                return result

        return Context()

    def finish(self, next_chunk):
        self.transient.clear()
        self.reads.clear()
        oldest = max(0, next_chunk - self.policy.history_chunks)
        for k in list(self.owned):
            if k.chunk < oldest:
                del self.owned[k]

    def close(self):
        if self.push is not None:
            self.push.close(abort=sys.exc_info()[0] is not None)
            self.push = None
        super().close()
