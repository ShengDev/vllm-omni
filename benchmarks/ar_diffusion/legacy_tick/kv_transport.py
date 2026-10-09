# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Physical KV transport, independent of version selection and page placement."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVTransport:
    """Physical KV transport policy.

    IPC is explicit, single-node CUDA only, and never silently falls back.
    Hybrid is explicit topology-aware transport for static horizontal pages:
    pages stay on CUDA IPC for local peers and use NCCL P2P for remote peers.
    It is intentionally a separate backend so an ``ipc`` experiment can never
    change transport class because the process group spans multiple hosts.
    round_overlap explicitly launches static hybrid rounds at Q/K/V publication
    on a separate stream, with readiness waits at the consuming history read.
    Mailbox IPC reserves (world - 1) * slots * slot_bytes plus small flags.
    The optional native kv-direct-slots path sizes final contexts from model/policy
    shapes instead; slots controls generations and slot_bytes is unused there.
    PreviousTickKV IPC uses schedule-sized slots (two per denoise source and
    H+1 per clean source). Its slots and slot_bytes options are unused.
    """

    backend: str = "p2p"
    slot_bytes: int = 32 * 1024**2
    slots: int = 2
    round_overlap: bool = False

    def __post_init__(self):
        if type(self.round_overlap) is not bool or (self.round_overlap and self.backend != "hybrid"):
            raise ValueError("round_overlap requires hybrid KV transport and a boolean value")
        if self.backend not in ("p2p", "ipc", "hybrid"):
            raise ValueError("KV transport backend must be p2p, ipc or hybrid")
        for name in ("slot_bytes", "slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


def fragments(transfers, slot_bytes):
    """Plan byte copies per directed edge without packing or gathering payloads.

    Each segment is (transfer index, field index, tensor offset, slot offset,
    byte count). Every rank derives the same plan from the agreed signatures.
    """
    import math

    import torch

    edges = {}
    for index, (_, src, dst, signature, _) in enumerate(transfers):
        if src == dst:
            continue
        rounds = edges.setdefault((src, dst), [[]])
        for field, (shape, dtype) in enumerate(signature):
            if shape is None:
                continue
            remaining = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            offset = 0
            while remaining:
                used = sum(segment[4] for segment in rounds[-1])
                if used == slot_bytes:
                    rounds.append([])
                    used = 0
                size = min(remaining, slot_bytes - used)
                rounds[-1].append((index, field, offset, used, size))
                offset += size
                remaining -= size
    return {edge: rounds for edge, rounds in edges.items() if rounds[0]}
