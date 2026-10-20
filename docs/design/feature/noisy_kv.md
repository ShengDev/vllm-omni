# Noisy KV

> **Status:** experimental (`vllm_omni.experimental.ar_diffusion`). Not a public
> HTTP schema.
>
> **Tracking:** [#102](https://github.com/JiusiServe/vllm-omni-project-manage/issues/102)
> (v0.31), quality [#103](https://github.com/JiusiServe/vllm-omni-project-manage/issues/103)
> (v0.32). PR [#8282](https://github.com/vllm-project/vllm-omni/pull/8282).

## Table of Contents

- [Overview](#overview)
  - [What is a cell?](#what-is-a-cell)
- [Scope](#scope)
- [Architecture](#architecture)
- [Cell schedule](#cell-schedule)
- [Cell executor](#cell-executor)
- [NoisyKV](#noisykv)
- [Configuration](#configuration)
- [Compatibility](#compatibility)
- [Testing](#testing)
- [Open issues](#open-issues)
- [Related](#related)

## Overview

Noisy KV is the same-request multi-chunk path on Omni's existing **layer PP**.

### What is a cell?

A **cell** `(c, s, b)` is one DiT **block** forward on one tick.

| | | |
| --- | --- | --- |
| `c` | chunk | Temporal latent slice |
| `s` | step | `0…D-1` noisy; `D` clean if present |
| `b` | block | Transformer block index |

History KV matches the **same** `b` only. With `K>1`, one `(c,s)` spans multiple ticks.

| Ordering | Advance | History KV |
| --- | --- | --- |
| **Self-Forcing** | One chunk at a time | Clean `(c', D, b)` only |
| **Latest-KV** | Diagonal | Newest finished `(c', s*, b)` |
| **Same-Step** | Same diagonal | Matching `(c', s, b)`; local only; **no clean** |

Vertical: `S=D` no clean; `S=D+1` with clean. CausVidWan: Self-Forcing / Latest-KV only.

#102: basic function + recorded config. No FPS SLA. DreamZero / LingBot defaults must not regress.

## Scope

| Axis | Unit |
| --- | --- |
| **Weight split** | Layer PP: `G = ⌈B/K⌉`; `pipeline_parallel_size = S·G` |
| **Schedule / KV** | Cell `(chunk, step, block)` |

`cell_schedule` picks the ordering for one `CellPlan` (not a session property).
Optional session keeps clean KV across generations
([Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)).

## Architecture

| Symbol | Config | Meaning |
| --- | --- | --- |
| `N` | `num_chunks` | Chunks |
| `D` | `num_denoise_steps` | Noisy steps only |
| `S` | `step_replicas` | Step replicas; vertical `S ∈ {1, D, D+1}` |
| `G` | `layer_groups` | Groups per replica; `world = S·G` |
| `B`, `K` | `blocks`, `blocks_per_rank` | DiT blocks / blocks per rank |
| `H` | `kv_history_chunks` | History window (not clean) |
| `has_clean_stage` | same | Clean@0 at step `D`; vertical `S = D + int(has_clean)` |
| `R` | `max_batch_size` | Micro-batch cap |

```text
s = pp_rank // G ,   g = pp_rank % G ,   rank(s, g) = s·G + g
```

Layer split uses `(g, G)` via `get_pp_indices`, not `world = S·G`.

```text
ARDiffusionModelRunner
  ├─ Legacy → paged KV (DreamZero / LingBot)
  └─ Cell   → NoisyKV + CellPlan → cell_executor
```

1. Build `CellPlan` from `extra_args`.
2. Bind `ARDiffusionCellContext` / `NoisyKVCache`.
3. Tick loop: compute + KV/activation P2P.
4. Rank 0 gathers or streams latents for VAE.

## Cell schedule

`experimental/ar_diffusion/cell_schedule.py`.

- **In:** `CellSchedule(N, D, S, G, B, K, ordering, H, has_clean_stage)`.
- **Out:** `CellPlan` — cells, `sources` / `transfers` / `last_use`.

```text
# Latest-KV / Self-Forcing
s* = max { s | P(c', s, b) < tick };  src = (c', s*, b)

# Same-Step (local only)
src = (c', s_task, b)  if P(c', s_task, b) < tick
```

Invariants: `S ∈ {1,D,D+1}`; Same-Step ⇒ `S=D`; Self-Forcing vertical ⇒ `S=D+1`;
no same-tick Latest edge; finite `H` + `last_use`.

## Cell executor

`experimental/ar_diffusion/cell_executor.py`.

Per tick: wait KV → one block forward → publish KV → post `isend` → wait/evict.
Received versions usable only on a later tick. `run_cell_pipeline()` owns admit /
fail-closed cleanup.

## NoisyKV

`experimental/ar_diffusion/kv_cache/noisy.py`.

| | |
| --- | --- |
| Unit | K/V for `(req, c, s, b)` |
| Storage | Per-layer `VersionPool` |
| Transport | Planned PP P2P on cell complete |
| Eviction | `last_use`; no overwrite of live sends |

## Configuration

| Layer | Key | Meaning |
| --- | --- | --- |
| `parallel_config` | `pipeline_parallel_size` | `S·G` |
| `ar_diffusion_stage_config` | `step_replicas` | `S` (alias `stage_parallel_size`) |
| | `max_batch_size` / `max_history_chunks` | `R` / `H` cap |
| `extra_args` | `cell_schedule` | CausVid: `self_forcing` \| `latest_kv` |
| | `num_chunks`, `num_denoise_steps`, `kv_history_chunks` | `N`, `D`, `H` |
| | `has_clean_stage` | Optional; else from `S` vs `D` (S=1: `H>0`) |

`source_rank` = last noisy step's trailing rank (`S=D` or `S=D+1`).

## Compatibility

| Combination | Status |
| --- | --- |
| No session + Self-Forcing / Latest-KV (CausVid) | Supported (opt-in) |
| Same-Step | Schedule only; not on CausVidWan |
| Session + legacy loop | Unchanged |
| Session + Cell | Designed; not required for #102 |

Capability: `SupportsARDiffusionCellPipeline`.

## Testing

CPU: `tests/diffusion/ar_diffusion/test_cell_{schedule,executor}.py`, `test_noisy_kv*.py`.

GPU: `S=1` Self-Forcing; `S=D+1` Self-Forcing vs Latest-KV.

## Open issues

- Paged session vs NoisyKV still separate; no TP × Noisy KV.
- Noisy KV not wired to session.
- Micro-batch packing (`R`).
- `K>1` multiplies ticks; activation inbox keyed by `(chunk, step)`.

## Related

- [Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)
- [AR-Diffusion pipeline capability](../ar_diffusion_pipeline_capability.md)
- [Pipeline Parallel](pipeline_parallel.md)
- Recipe: [`recipes/Physis-AI/CausVid-Wan.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Physis-AI/CausVid-Wan.md)
- Code: `experimental/ar_diffusion/{cell_schedule,cell_executor,kv_cache/noisy}.py`
