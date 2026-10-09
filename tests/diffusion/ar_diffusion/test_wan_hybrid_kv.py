# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native KV visibility and physical-address contracts for transport adapters."""

import pytest
import torch

from benchmarks.ar_diffusion.hybrid_kv import LayerMajorPages, NativeTickManager, validate_native_plan
from benchmarks.ar_diffusion.legacy_tick.cell_schedule import Vertical
from benchmarks.ar_diffusion.legacy_tick.kv_policy import KVKey
from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    ChunkSchedule,
    Ordering,
    build_chunk_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("steps", [4, 5])
@pytest.mark.parametrize("chunks", [1, 7, 128])
def test_block_plan_preserves_native_versions_and_active_receivers(groups, steps, chunks):
    plan = build_chunk_plan(ChunkSchedule(chunks, steps, steps + 1, groups, Ordering.INTERLEAVED, 6))
    validate_native_plan(plan, Vertical(30 // groups).schedule(30, steps + 1, chunks))


def test_ipc_addresses_point_to_the_contiguous_attention_pools():
    pages = object.__new__(LayerMajorPages)
    pages.rank, pages.capacity, pages.stages, pages.storage_blocks = 0, 7, 5, 3
    pages.manager = object.__new__(NativeTickManager)
    pages.manager.block_slots = [{10: 0, 11: 1, 12: 2}]
    pages.buffers = torch.empty((3, 2, 7, 5, 1, 8, 2, 4), dtype=torch.bfloat16)
    pages.bytes_per_tensor = 8 * 2 * 4 * pages.buffers.element_size()
    for chunk in (0, 6, 7, 14, 127):
        for step in range(5):
            for block in (10, 11, 12):
                key = KVKey(chunk, step, block)
                for field, value in enumerate(pages.page(key)):
                    address = pages.address(pages.buffers.data_ptr(), key, 0, field)
                    assert value.data_ptr() == address
                    pool = pages.buffers[block - 10, field].view(-1, 2, 4)
                    slot = chunk % 7 * 5 + step
                    assert pool[slot * 8 : (slot + 1) * 8].data_ptr() == address
                    assert value.is_contiguous() and pool.is_contiguous()
