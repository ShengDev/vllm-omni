# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Clean-page placement policy; node identity is explicit, not rank arithmetic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVStorage:
    clean_placement: str = "producer"
    prefer_same_node: bool = True
    allow_cross_node: bool = False
    clean_budget_bytes: int | None = None  # Per rank, authoritative clean pages only.
    replica_cache_bytes: int = 0  # Per rank, bounded remote clean AND noisy cache.
    node_ids: tuple[str, ...] | None = None  # Group-rank order; otherwise detect hostnames.

    clean_ranks: tuple[int, ...] | None = None  # Optional storage candidate pool.

    def __post_init__(self):
        if self.clean_placement not in ("producer", "balanced"):
            raise ValueError("clean_placement must be producer or balanced")
        if type(self.prefer_same_node) is not bool or type(self.allow_cross_node) is not bool:
            raise ValueError("node preferences must be boolean")
        for name in ("clean_budget_bytes", "replica_cache_bytes"):
            n = getattr(self, name)
            if n is None and name == "clean_budget_bytes":
                continue
            if type(n) is not int or n < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.clean_ranks is not None:
            if (
                self.clean_placement != "balanced"
                or not isinstance(self.clean_ranks, tuple)
                or not self.clean_ranks
                or len(set(self.clean_ranks)) != len(self.clean_ranks)
                or any(type(r) is not int or r < 0 for r in self.clean_ranks)
            ):
                raise ValueError("clean_ranks requires balanced placement and distinct nonnegative ranks")
        if self.node_ids is not None and (
            not isinstance(self.node_ids, tuple)
            or not self.node_ids
            or any(not isinstance(n, str) or not n for n in self.node_ids)
        ):
            raise ValueError("node_ids must be a nonempty tuple of node names")

    def owner(self, producer, size, nodes, used):
        if self.clean_placement == "producer":
            candidates = [producer]
        else:
            candidates = [
                r
                for r in (self.clean_ranks if self.clean_ranks is not None else range(len(nodes)))
                if self.allow_cross_node or nodes[r] == nodes[producer]
            ]
        candidates = [
            r for r in candidates if self.clean_budget_bytes is None or used[r] + size <= self.clean_budget_bytes
        ]
        if not candidates:
            raise MemoryError(
                "clean KV budget exhausted on eligible ranks; increase budget, reduce history, "
                "or explicitly enable cross-node placement"
            )
        return min(candidates, key=lambda r: (int(self.prefer_same_node and nodes[r] != nodes[producer]), used[r], r))
