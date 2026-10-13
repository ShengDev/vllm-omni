# Noisy PP

> **Status:** experimental (`vllm_omni.experimental.ar_diffusion`). Not a public
> HTTP schema.
>
> **Tracking:** [#102](https://github.com/JiusiServe/vllm-omni-project-manage/issues/102)
> (v0.31), quality [#103](https://github.com/JiusiServe/vllm-omni-project-manage/issues/103)
> (v0.32). PR [#8282](https://github.com/vllm-project/vllm-omni/pull/8282).

## Scope

Same-request multi-chunk scheduling on Omni's existing **layer PP**.

| Axis | Unit |
| --- | --- |
| **Model / weight split** | Unchanged PP: ranks own contiguous layer groups (`G = ⌈B/K⌉`); `pipeline_parallel_size = S·G` |
| **Schedule / KV** | **Block (cell)** `(chunk, step, block)` — one transformer block per tick |

1. **Latest** (`Ordering.INTERLEAVED`): diagonal overlap; history reads newest finished `(c', s*, b)` with completion tick strictly earlier.
2. **Serial** (`Ordering.SERIAL`): finish one chunk before the next; history reads clean only. Same topology; Self-Forcing baseline.

#102 acceptance: reproducible basic function + recorded config. No FPS / speedup SLA. DreamZero / LingBot paged-session defaults must not regress.

## Overview

- `S = 1`: native layer PP.
- `S = T+1`: one weight replica per denoise step + clean stage.

Activations still follow PP (`next_rank` after the block that feeds the next group).
KV is keyed **per block** `b`: publish / transfer / wait when that cell finishes;
a remote reader of `b` does not wait for the producer's other local blocks.

| Strategy | Advance | History KV |
| --- | --- | --- |
| Serial | All cells of a chunk, then next chunk | Clean `(c', T, b)` |
| Latest | Diagonal cell overlap | Newest finished `(c', s*, b)` |

`chunk_schedule` is per generation on one `ChunkPlan`, not a session property.
Optional session only keeps clean KV across generations
([realtime_ar_diffusion](realtime_ar_diffusion.md)).

## Architecture

| Symbol | Meaning |
| --- | --- |
| `N`, `T` | Chunks; denoise steps (`T` = clean forward) |
| `B`, `K` | Blocks; contiguous blocks per rank in a stage |
| `S`, `G` | Stages; groups per stage. Only `S ∈ {1, T+1}` |
| Cell `(c,s,b)` | Schedule task + KV version (+ `req` when batched) |
| Tick | One cell on one rank (one block forward) |
| `R` | Max micro-batch requests per rank per tick |

```text
s = pp_rank // G ,   g = pp_rank % G ,   rank(s, g) = s·G + g
```

Layer split uses `(g, G)` via `get_pp_indices(B, pp_rank % G, G)`, not
`world = S·G`. Production Wan2.2 PP stays on its existing path.

```text
Optional Session
        │
        ▼
ARDiffusionModelRunner
  ├─ Legacy → paged KV (DreamZero / LingBot)
  └─ Chunk  → NoisyKV + ChunkPlan
        │
   chunk_schedule │ chunk_executor (tick loop, PP P2P)
        └────► kv_cache/noisy.py
```

Architecture diagrams (two stacks, call chain, KV peers, target layout) live under
[`figs/`](figs/) and are embedded in the Chinese design note
[`noisy_pp.zh.md`](noisy_pp.zh.md) (§2.3).

### Tick loop

1. Wait transfers needed by this cell's `sources` (same `b`).
2. Forward block; **publish KV immediately**.
3. Post planned KV `isend` (no wait for remote recv).
4. Wait posted handles; `evict` versions past last-use with sends done.

Received versions readable only on a later tick. Same-tick producer/consumer of
the same version is forbidden.

## Scheduling

`experimental/ar_diffusion/chunk_schedule.py` (target).

- In: `ChunkSchedule(N, T, S, G, B, K, ordering, H)`.
- Out: `ChunkPlan` — per-tick cells `(c,s,b)`, plus `sources` / `transfers` /
  `last_use` keyed by cell.

Visibility (same `b`):

```text
s* = max { s | P(c', s, b) < current_tick }
src = (c', s*, b)
```

Invariants: `S ∈ {1, T+1}`; no Latest edge in the same micro-batch/tick;
cross-rank KV ready before use; finite `H` + last-use bounds residency;
consumer of `b` never waits on producer's `b' ≠ b` (activation edges separate).

## NoisyKV

`experimental/ar_diffusion/kv_cache/noisy.py` (target).

| | |
| --- | --- |
| Unit | K/V for `(req, c, s, b)` — one block |
| Storage | Per-layer `VersionPool` from `ARDiffusionNoisyKVSpec` |
| Transport | Planned PP-group P2P; post on cell complete |
| Eviction | `last_use` per cell; no in-place overwrite of live sends |

Per-tick KV traffic ≈ history depth × one block × tokens × heads × dim per
remote source. Mitigations: post-on-produce, dedupe `(version, dst)`, larger `G`.

## Compatibility

Only-`SupportsARDiffusionPipeline` paths keep today's behavior. Default
DreamZero / LingBot YAMLs must not silently enter `run_chunk_pipeline`.

| Combination | Status |
| --- | --- |
| No session + Serial/Latest | Supported (opt-in) |
| Session + legacy loop | Supported (unchanged) |
| Session + Chunk | Designed; not required for #102 |

Capability: `SupportsARDiffusionChunkPipeline` —
`ar_diffusion_noisy_kv_spec()`, `bind_ar_diffusion_chunk_context()`.

## Configuration

| Layer | Key | Meaning |
| --- | --- | --- |
| `parallel_config` | `pipeline_parallel_size` | `S·G` |
| `ar_diffusion_stage_config` | `stage_parallel_size` | `S` |
| | `max_batch_size` | `R` |
| | `max_history_chunks` | Cap for `H` |
| Request `extra_args` | `chunk_schedule` | `serial` \| `latest` |
| | `num_chunks`, `num_denoise_steps`, `kv_history_chunks` | `N`, `T`, `H` |

Backend: `ARDiffusionEngine` + mp executor from deploy YAML. For `S > 1`,
`S = num_denoise_steps + 1`.

## Testing

CPU: `tests/diffusion/ar_diffusion/test_chunk_{schedule,executor}.py`,
`test_noisy_kv*.py`, model/config resolve tests under the same tree.

GPU (record config): `S=1` serial; `S=2` (`T=1`); `S=T+1` Serial vs Latest
(`benchmarks/noisy_pp/`).

## Open issues

- Paged session vs NoisyKV still separate prealloc; no TP × Noisy PP yet.
- Noisy PP not wired to session.
- Micro-batch packing (`R`): varlen vs pad.
- Cell schedule with `K>1` multiplies ticks; activation inbox is keyed by
  `(chunk, step)` (R=1 friendly).

## Related

- [Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)
- [AR-Diffusion pipeline capability](../ar_diffusion_pipeline_capability.md)
- [Pipeline Parallel](pipeline_parallel.md)
- Code: `experimental/ar_diffusion/{chunk_schedule,chunk_executor,kv_cache/noisy}.py`
