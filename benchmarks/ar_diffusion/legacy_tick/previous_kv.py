# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Full-DiT Same Step lanes reading only previous-round KV versions."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreviousTickKV:
    """R10: latest completed version strictly before the query's block tick.

    History counts previous chunks, excludes self, and has no sink/rebasing.
    Each lane owns a full DiT; new pages become visible next model round.
    """

    history_chunks: int = 6
    dependency = "previous_tick"
    clean_store = True
    sink_chunks = 0
    rebase_sink = False

    def __post_init__(self):
        if type(self.history_chunks) is not int or self.history_chunks < 0:
            raise ValueError("history_chunks must be a nonnegative integer")

    def fork(self):
        return PreviousTickCache(self.history_chunks)


class PreviousTickCache:
    def __init__(self, history):
        self.history = history
        self.pages, self.pending, self.rings = {}, {}, {}
        self.reads = None

    def context(
        self, step, block, chunk, key, value, *, unrotated_key=None, rotate=None, token_major=False, ring=False
    ):
        cache = self.pages.setdefault(block, {})
        labels, parts = [], []
        for c in range(max(0, chunk - self.history), chunk):
            version, k, v = cache[c]
            if c + version >= chunk + step:
                raise RuntimeError("previous-tick KV exposed a current/future version")
            labels.append((c, version, block))
            parts.append((k, v))
        if self.reads is not None:
            self.reads.append((chunk, step, block, tuple(labels)))
        parts.append((key, value))
        self.stage(block, key, value)
        axis = 1 if token_major else 2

        def convert(x):
            return x.transpose(1, 2) if token_major else x

        if ring:
            # Versioned mirrored slots: only changed versions are copied. Current
            # writes never replace pages required by this chronological window.
            count, n = self.history + 1, key.shape[2]
            signature = (tuple(key.shape), key.dtype, key.device, token_major)
            if block not in self.rings:
                shape = list(convert(key).shape)
                shape[axis] = 2 * count * n
                self.rings[block] = (signature, [key.new_empty(shape), value.new_empty(shape)], {})
            previous, buffers, versions = self.rings[block]
            if previous != signature:
                raise ValueError("KV ring signature changed during request")
            ids = [(c, version) for c, version, _ in labels] + [(chunk, step)]
            for (c, version), pair in zip(ids, parts):
                slot = c % count
                if versions.get(slot) != (c, version):
                    for buffer, x in zip(buffers, pair):
                        buffer.narrow(axis, slot * n, n).copy_(convert(x))
                        buffer.narrow(axis, (slot + count) * n, n).copy_(convert(x))
                    versions[slot] = (c, version)
            start = ((chunk % count) + count + 1 - len(parts)) * n
            result = tuple(buffer.narrow(axis, start, len(parts) * n) for buffer in buffers)
        else:
            result = tuple(torch.cat([convert(p[i]) for p in parts], dim=axis) for i in (0, 1))
        return tuple(x.transpose(1, 2) for x in result) if token_major else result

    def stage(self, block, key, value):
        self.pending[block] = (key.detach().clone(), value.detach().clone())

    def commit(self, block, chunk, version, key, value):
        self.pages.setdefault(block, {})[chunk] = (version, key, value)

    def prune(self, next_chunk, chunks):
        for block, pages in self.pages.items():
            self.pages[block] = {
                c: p
                for c, p in pages.items()
                if 0 <= next_chunk < chunks and max(0, next_chunk - self.history) <= c < next_chunk
            }
        self.pending.clear()
