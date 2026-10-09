# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""KV version selection, independent of execution layout and physical storage."""

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class KVKey:
    chunk: int
    step: int
    block: int


@dataclass(frozen=True)
class KVPolicy:
    """Use with Engine: layout is configured separately in Execution.

    history_chunks=None retains unlimited history. Latest reads a snapshot of
    versions completed before this block tick, never publication arrival order.
    """

    selection: str = "same_step"
    history_chunks: int | None = 6
    sink_chunks: int = 0
    rebase_sink: bool = False
    clean_store: bool = False
    dependency = "managed"

    def __post_init__(self):
        if self.selection not in ("same_step", "latest", "clean"):
            raise ValueError("selection must be same_step, latest or clean")
        if self.history_chunks is not None and (type(self.history_chunks) is not int or self.history_chunks < 0):
            raise ValueError("history_chunks must be nonnegative or None")
        if type(self.sink_chunks) is not int or self.sink_chunks < 0:
            raise ValueError("sink_chunks must be nonnegative")
        if type(self.clean_store) is not bool or type(self.rebase_sink) is not bool:
            raise ValueError("clean_store and rebase_sink must be boolean")
        if self.clean_store and self.selection not in ("latest", "clean"):
            raise ValueError("clean_store requires latest or clean selection")
        if self.selection == "clean" and not self.clean_store:
            raise ValueError("clean selection requires clean_store")
        if self.rebase_sink and self.sink_chunks != 1:
            raise ValueError("rebasing requires one sink chunk")

    def history(self, chunk):
        start = self.sink_chunks if self.history_chunks is None else max(self.sink_chunks, chunk - self.history_chunks)
        return (*range(min(chunk, self.sink_chunks)), *range(start, chunk))

    def select(self, task, directory):
        keys = []
        for chunk in self.history(task.chunk):
            if self.selection == "same_step":
                key = KVKey(chunk, task.step, task.block)
            else:
                candidates = [
                    key
                    for key in directory
                    if key.chunk == chunk
                    and key.block == task.block
                    and (self.selection != "clean" or directory[key].clean)
                ]
                if not candidates:
                    raise ValueError(f"KV not ready: chunk={chunk}, block={task.block}")
                key = max(candidates, key=lambda k: k.step)
            if key not in directory:
                raise ValueError(f"KV not ready: {key}")
            keys.append(key)
        return tuple(keys)


@dataclass(frozen=True)
class SelfForcingKV(KVPolicy):
    """Serial chunk rollout with mandatory t=0 refresh and clean-only history.

    Engine defaults to Execution(layout="serial") for this policy. All denoise
    steps and the clean pass reuse the same block partitions and weights.
    """

    selection: str = field(default="clean", init=False)
    clean_store: bool = field(default=True, init=False)
