# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Playground VL: schedule first, select the latest STRICTLY earlier tick.

K is part of the numerical contract. This is not PreviousTickKV's frozen K=B
read set. Physical arrival order never chooses a version.
"""

from .horizontal_kv_ipc import PushPages
from .kv_policy import KVKey
from .managed import schedule as fifo_schedule
from .vertical_kv import StaticVerticalLatest, VerticalRoundPages


def schedule(blocks, stages, chunks, partitions, partition_major=False):
    count = len(partitions)
    order = (
        tuple(part * stages + step for step in range(stages) for part in range(count))
        if partition_major
        else tuple(range(stages * count))
    )
    return fifo_schedule(blocks, stages, chunks, partitions, order)


class TickPlan:
    def __init__(self, tasks, stages, history):
        if history is None or history < 0:
            raise ValueError("static tick-latest requires finite nonnegative history")
        self.tasks = tuple(sorted(tasks, key=lambda t: (t.tick, t.rank)))
        self.by_key = {KVKey(t.chunk, t.step, t.block): t for t in self.tasks}
        if len(self.by_key) != len(self.tasks):
            raise ValueError("duplicate tick-latest task")
        self.owners, self.reads, self.destinations = {}, {}, {}
        for task in self.tasks:
            identity = task.step, task.block
            if self.owners.setdefault(identity, task.rank) != task.rank:
                raise ValueError("tick-latest requires fixed step/block ownership")
            selected = []
            for chunk in range(max(0, task.chunk - history), task.chunk):
                for step in range(stages - 1, -1, -1):
                    key = KVKey(chunk, step, task.block)
                    producer = self.by_key.get(key)
                    if producer is not None and producer.tick < task.tick:
                        selected.append(key)
                        # Sorted task order makes this the last reader on this rank.
                        self.destinations.setdefault(key, {})[task.rank] = (task.chunk, task.step)
                        break
                else:
                    raise ValueError(f"no earlier-tick history for {task}, chunk {chunk}")
            self.reads[task] = tuple(selected)

    def local_only(self, hosts):
        return all(
            hosts[self.owners[key.step, key.block]] == hosts[rank]
            for key, readers in self.destinations.items()
            for rank in readers
        )


class TickPages(VerticalRoundPages):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use actual IPC handshake hostnames, never assumed node IDs.
        self.local_only = self.manager.plan.local_only(self.hosts)

    def plan_destinations(self, key):
        return self.manager.plan.destinations.get(key, {})

    def prepare_round(self, tasks):
        if not self.local_only:
            super().prepare_round(tasks)

    def publish(self, key, k, v):
        if self.local_only:
            # The IPC path already owns readiness, release and allocator fences.
            # No cross-host operations exist, so do not construct an empty round.
            return PushPages.publish(self, key, k, v)
        return super().publish(key, k, v)

    def commit_round(self):
        if not self.local_only:
            return super().commit_round()


class StaticTickLatest(StaticVerticalLatest):
    page_type = TickPages

    def __init__(self, policy, storage, nodes, stages, *args, tasks, **kwargs):
        self.plan = TickPlan(tasks, stages, policy.history_chunks)
        super().__init__(policy, storage, nodes, stages, *args, tasks=tasks, **kwargs)

    def owner(self, step, block):
        return self.plan.owners[step, block]

    def keys_for(self, task):
        return self.plan.reads[task]
