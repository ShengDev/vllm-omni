# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Balanced physical VL placement, preserving the frozen logical VL read set.

Logical tick is still (chunk + step)*blocks + block. Physical ticks only
schedule that DAG on a chosen number of GPUs; arrival never selects versions.
"""

from functools import lru_cache

from .horizontal_kv import StaticSameStep
from .horizontal_kv_round import RoundPushPages
from .kv_policy import KVKey, KVPolicy
from .managed import Task
from .previous_push import readers


def owner(step, block, blocks, stages, world, partition_major=False):
    rank = (step * blocks + block) * world // (stages * blocks)
    if partition_major:
        if world % stages or blocks % (world // stages):
            raise ValueError("partition-major VL requires equal per-stage block partitions")
        parts = world // stages
        rank = (rank % parts) * stages + rank // parts
    return rank


def keys(chunk, step, block, history, stages):
    return tuple(KVKey(c, min(stages - 1, chunk + step - 1 - c), block) for c in range(max(0, chunk - history), chunk))


def schedule(blocks, stages, chunks, world, history, partition_major=False):
    if not 1 <= world <= stages * blocks:
        raise ValueError("vertical_ranks must be between 1 and stages*blocks")
    occupied, lookup, tasks = [-1] * world, {}, []
    for chunk in range(chunks):
        predecessor = -1
        for step in range(stages):
            for block in range(blocks):
                rank = owner(step, block, blocks, stages, world, partition_major)
                dependencies = keys(chunk, step, block, history, stages)
                tick = 1 + max(predecessor, occupied[rank], max((lookup[key] for key in dependencies), default=-1))
                task = Task(tick, rank, chunk, step, block)
                tasks.append(task)
                lookup[KVKey(chunk, step, block)] = tick
                occupied[rank] = predecessor = tick
    return sorted(tasks, key=lambda task: (task.tick, task.rank))


class VerticalRoundPages(RoundPushPages):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._destinations = lru_cache(maxsize=2 * self.capacity * self.stages * self.blocks)(self.plan_destinations)

    def source_rank(self, key):
        return self.manager.owner(key.step, key.block)

    def storage_index(self, key, rank=None):
        return self.manager.block_slots[self.rank if rank is None else rank][key.block]

    def all_destinations(self, key):
        return self._destinations(key)

    def plan_destinations(self, key):
        result, last_tick = {}, {}
        for step in range(self.stages):
            interval = readers(key.chunk, key.step, step, self.chunks, self.history, self.stages)
            if interval is None:
                continue
            chunk = interval[1]
            destination = self.manager.owner(step, key.block)
            tick = self.manager.task_ticks[chunk, step, key.block]
            if tick > last_tick.get(destination, -1):
                result[destination] = (chunk, step)
                last_tick[destination] = tick
        return result

    def reader_identity(self, chunk, step):
        return chunk, step


class StaticVerticalLatest(StaticSameStep):
    page_type = VerticalRoundPages

    def __init__(self, policy, *args, tasks, partition_major=False, **kwargs):
        self.partition_major = partition_major
        self.task_ticks = {(t.chunk, t.step, t.block): t.tick for t in tasks}
        super().__init__(
            KVPolicy(selection="latest", history_chunks=policy.history_chunks, clean_store=True), *args, **kwargs
        )

    def owner(self, step, block):
        return owner(step, block, self.blocks, self.stages, len(self.nodes), self.partition_major)

    def make_push(self):
        world = len(self.nodes)
        self.block_slots = []
        for rank in range(world):
            needed = sorted({b for s in range(self.stages) for b in range(self.blocks) if self.owner(s, b) == rank})
            self.block_slots.append({block: slot for slot, block in enumerate(needed)})
        # Equal padded allocations keep IPC metadata uniform while each rank
        # stores only block positions it can query (12 of 30 for N16/B30).
        self.storage_blocks = max(map(len, self.block_slots))
        return self.page_type(self, self.shape, self.dtype)

    def keys_for(self, task):
        return keys(task.chunk, task.step, task.block, self.policy.history_chunks, self.stages)

    def prepare(self, tasks):
        if self.push is not None:
            return super().prepare(tasks)
        self.transient, self.reads = {}, {}
        transfers = []
        for task in tasks:
            self.reads[task] = self.keys_for(task)
            for key in self.reads[task]:
                source = self.owner(key.step, key.block)
                transfers.append(
                    (
                        (task.rank, key),
                        source,
                        task.rank,
                        self.signature,
                        self.owned.get(key) if self.local(source) else None,
                    )
                )
                if source != task.rank:
                    self.sent_bytes[source] += self.page_bytes
                    self.received_bytes[task.rank] += self.page_bytes
        self.transient = self.transfer(transfers, kv=True)
