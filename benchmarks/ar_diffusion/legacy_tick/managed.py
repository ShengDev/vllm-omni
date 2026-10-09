# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Pure scheduling compatibility; no WaveServe model execution."""

from .cell_schedule import Task, Vertical


def schedule(blocks, stages, chunks, partitions, order):
    base = Vertical(len(partitions[0])).schedule(blocks, stages, chunks)
    return [Task(t.tick, order[t.rank], t.chunk, t.step, t.block) for t in base.tasks]
