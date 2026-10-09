# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Block scheduling: which rank computes which cell (chunk, step, block) at which tick.

One tick is one forward pass of one block. A layout fixes the rank topology and the tick of every
cell and knows nothing about KV; the two layouts are the two partitionings of the design note.

Vertical: rank = (stage, block group). Each rank holds K contiguous blocks of one stage; chunks flow
through the ranks as a pipeline, first-come first-served on each rank.
Horizontal: every rank holds the full DiT; chunk c runs all of its stages on rank c % N.
"""

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Task:
    tick: int
    rank: int
    chunk: int
    step: int
    block: int

    @property
    def key(self) -> tuple[int, int, int]:
        """(chunk, step, block): the cell's identity and the key of the KV page it writes."""
        return self.chunk, self.step, self.block


class Schedule:
    """Every cell of a request placed on (tick, rank); identical on every rank."""

    def __init__(self, tasks: list[Task], blocks: int, stages: int, ranks: int):
        self.tasks = sorted(tasks)
        self.blocks, self.stages, self.ranks = blocks, stages, ranks
        self.chunks = self.tasks[-1].chunk + 1
        self.task_of = {t.key: t for t in self.tasks}
        self.ticks = self.tasks[-1].tick + 1
        self.by_tick: list[list[Task]] = [[] for _ in range(self.ticks)]
        for task in self.tasks:
            self.by_tick[task.tick].append(task)
        if any(len({t.rank for t in batch}) != len(batch) for batch in self.by_tick):
            raise AssertionError("two cells scheduled on one rank in the same tick")

    def task(self, chunk: int, step: int, block: int) -> Task:
        return self.task_of[chunk, step, block]

    def output(self, chunk: int, steps: int) -> Task:
        """The cell after which chunk's latent is final: the last block of the last denoising pass."""
        return self.task(chunk, steps - 1, self.blocks - 1)

    def successor(self, task: Task) -> Task | None:
        """The cell that consumes this cell's output: next block, else block 0 of the next stage."""
        if task.block + 1 < self.blocks:
            return self.task(task.chunk, task.step, task.block + 1)
        if task.step + 1 < self.stages:
            return self.task(task.chunk, task.step + 1, 0)
        return None


class Vertical:
    def __init__(self, blocks_per_rank: int | None = None):
        if blocks_per_rank is not None and (type(blocks_per_rank) is not int or blocks_per_rank < 1):
            raise ValueError("blocks_per_rank must be a positive integer or None (whole DiT per rank)")
        self.blocks_per_rank = blocks_per_rank

    def partitions(self, blocks: int) -> tuple[range, ...]:
        k = self.blocks_per_rank or blocks
        return tuple(range(a, min(a + k, blocks)) for a in range(0, blocks, k))

    def num_ranks(self, blocks: int, stages: int) -> int:
        return stages * len(self.partitions(blocks))

    def rank_blocks(self, rank: int, blocks: int, stages: int) -> range:
        parts = self.partitions(blocks)
        if not 0 <= rank < stages * len(parts):
            raise ValueError(f"rank {rank} is outside the {stages * len(parts)} ranks of this layout")
        return parts[rank % len(parts)]

    def schedule(self, blocks: int, stages: int, chunks: int) -> Schedule:
        parts = self.partitions(blocks)
        ends = [-1] * (stages * len(parts))
        tasks = []
        for chunk in range(chunks):
            previous = -1
            for step in range(stages):
                for group, part in enumerate(parts):
                    rank = step * len(parts) + group
                    start = max(previous + 1, ends[rank] + 1)
                    tasks.extend(Task(start + j, rank, chunk, step, block) for j, block in enumerate(part))
                    previous = ends[rank] = start + len(part) - 1
        return Schedule(tasks, blocks, stages, len(ends))


class Horizontal:
    def __init__(self, ranks: int | None = None):
        if ranks is not None and (type(ranks) is not int or ranks < 1):
            raise ValueError("ranks must be a positive integer or None (stages x blocks)")
        self.ranks = ranks

    def num_ranks(self, blocks: int, stages: int) -> int:
        return self.ranks if self.ranks is not None else stages * blocks

    def rank_blocks(self, rank: int, blocks: int, stages: int) -> range:
        if not 0 <= rank < self.num_ranks(blocks, stages):
            raise ValueError(f"rank {rank} is outside the {self.num_ranks(blocks, stages)} ranks of this layout")
        return range(blocks)

    def schedule(self, blocks: int, stages: int, chunks: int) -> Schedule:
        ranks, length = self.num_ranks(blocks, stages), stages * blocks
        tasks = []
        for chunk in range(chunks):
            rank = chunk % ranks
            start = (chunk // ranks) * max(length, ranks) + rank  # rounds of N chunks start one tick apart
            tasks.extend(
                Task(start + step * blocks + block, rank, chunk, step, block)
                for step in range(stages)
                for block in range(blocks)
            )
        return Schedule(tasks, blocks, stages, ranks)
