# H800: native Wan chunk IPC/NCCL optimization

An incremental benchmark on PR #8282 head
`86490babe358740cf98f019d1972f3b364a329d4` measures **129.714 FPS** on one node
and **251.514 FPS** on two nodes with the optimized adapter, a **1.939x**
scaling ratio. Within each topology, baseline and optimized full latents are
bitwise identical. These are steady pure-DiT pixel-frame equivalents; VAE and
the serving frontend are excluded.

| Native vLLM-Omni variant | One node, 5 DiT GPUs | Two nodes, 10 DiT GPUs | Scaling |
| --- | ---: | ---: | ---: |
| Unoptimized | 116.198 FPS | 178.486 FPS | 1.536x |
| Per-block IPC/NCCL + exact BF16 fusions | 129.714 FPS | 251.514 FPS | 1.939x |
| Improvement at the same topology | +11.63% | +40.91% | |

| Topology / variant | Three formal FPS samples | Maximum per-rank peak allocated memory |
| --- | --- | ---: |
| One node / baseline | 116.197775, 116.297489, 116.065247 | 14.866 GiB |
| One node / optimized | 129.714088, 129.524997, 129.744626 | 40.145 GiB |
| Two nodes / baseline | 178.486287, 178.519997, 178.343932 | 6.703 GiB |
| Two nodes / optimized | 251.410771, 251.513610, 252.225687 | 20.302 GiB |

The hybrid adapter uses 35 version positions per layer in addition to the
original native pool. It trades more GPU memory for overlap. Peak allocated
memory is reset before each variant; the table takes the maximum across
all ranks, including the full warmup. It is not `nvidia-smi` reserved memory.

## Hardware, software and configuration

Both nodes have eight NVIDIA H800 80 GB GPUs, a 700 W power limit and maximum
SM clock of 1980 MHz. Each benchmark uses five GPUs per node. Local traffic
uses NVLink; cross-node traffic uses NCCL RDMA/GDR. The single-node run uses
the second node; the two-node run uses both. Jobs run sequentially on the
same allocation.

Python 3.12.13, vLLM 0.29.0, PyTorch 2.13.0+cu130, NCCL 2.29.7+cuda13.2,
Diffusers 0.38.0, Triton 3.7.1, `cuda.bindings` 13.4.3 and driver 580.95.05.
Native chunk/model execution runs in this environment. Repository-wide test
plugins require a newer compatible vLLM API and cannot load here; the
targeted unit/kernel checks use `--noconftest`.

The model is Wan 2.1 1.3B RF, with 30 transformer blocks, 12 heads and head
size 128. Transformer weights SHA256:
`00638d589af492beadfd8ecb4d8e4ed9e5c4e4ad808b29cecc8618000272c9ce`.
Four RF Flow Euler denoise steps (`shift=5`) plus clean give five stages.
Stage-major placement uses `K=30` on one node and `K=15` on two nodes.

Batch one, BF16, seed zero, prompt `a cat walking on grass`, q=3, history six
chunks and no sink. Each complete request contains 128 chunks, three latent
frames per chunk, spatial latent size 60x104 and complete shape
`[1,16,384,60,104]`. Full VAE decoding would yield 1533 frames at 832x480;
decoding is not performed by this benchmark.

The final denoise rank records chunk completion CUDA events. The window is
skip64/measure32/tail32; 384 pixel-frame equivalents are divided by
`completion_ms[95] - completion_ms[63]`. T5 encoding, model loading, VAE,
hashing and video encoding are excluded. Each variant runs one full warmup
and three complete measured requests, reported as the median. See the
[harness documentation](../../README.md) for exact torchrun commands.

## Validation and evidence

All 16 full requests pass their topology's complete latent hash:

- K30: `d37a9c999e612e6578aad0ff8fd715ba3d5abfa4a46994f698adb3a8df9396fe`.
- K15: `a9a53546cb02479643554cce66ee217c7c00d84af6cbb16a0ecde54b60c2f45d`.

K-dependent Latest KV labels differ, so cross-topology equality is not
required. No cross-framework quality, concurrent-request, VAE or end-to-end
speedup is claimed. The harness validates its archived block schedule against
the native plan's KV labels, completion slots and active receivers.

The original native version-pool transport remains the baseline. The optimized
path publishes each block's KV as soon as Q/K/V are ready, using IPC push on
the same host and static NCCL rounds across hosts. Producer-ready and
consumer-release tickets protect slot reuse. Layer-major rings preserve
contiguous native paged attention pools. Pointwise kernels preserve every
intermediate eager BF16 rounding, with FP32 fusion disabled. Request caches
are cleared between requests, and context managers restore original modules.

Targeted checks on the PR head: **117 passed, 15 warnings in 20.84s**
(114 CPU checks and three CUDA BF16 checks):

```bash
python -m pytest --noconftest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_hybrid_kv.py \
  tests/diffusion/ar_diffusion/test_wan_native_pointwise.py \
  tests/diffusion/ar_diffusion/test_wan_native_optimizations.py \
  tests/diffusion/ar_diffusion/test_chunk_schedule.py \
  tests/diffusion/ar_diffusion/test_chunk_executor.py \
  tests/diffusion/ar_diffusion/test_noisy_kv.py \
  tests/diffusion/ar_diffusion/test_waveserve_chunk_layers.py -q
```

Ruff 0.14.10 check/format, Python AST parsing, SPDX, forbidden-import,
`torch.cuda` and test-mark gates, Python 3.10 mypy for the three added test
files, whitespace/debug hooks and Markdown lint were checked separately.
The complete pre-commit launcher failed while fetching hooks from GitHub;
it is not reported as passing. Full repository CI and frontend/end-to-end
tests have not been run.

[evidence.zip](evidence.zip) contains per-request completion events, hashes,
rank summaries, the targeted test log and measured source snapshots.
SHA256: `6e021eac5aa5d2196e99f424a7cf0d296d595b94434f11377eeab20f76047e2a`.
The audit recomputes all 16 requests and medians and checks 52 source snapshot
hashes. [audit.json](audit.json) also records memory and sample values.

```bash
python benchmarks/ar_diffusion/audit_results.py \
  benchmarks/ar_diffusion/results/h800-q3-20261009/evidence.zip
```

The fixture [conditioning.pt.gz](conditioning.pt.gz) contains pre-encoded
text only. Its uncompressed SHA256 is
`198358abb9eb6e80296d18bb78f82924a4fef1fa1a44ff26af34e75cfda9c5af`.
Model weights are not included. Transport source and adaptation hashes are
recorded in [provenance.json](../../legacy_tick/provenance.json).

For comparison, the earlier frozen commit
`2e4f22a3e3d67319fafbf84e1d0f199e397e08e9` measured 116.303/130.038 FPS on
one node and 178.097/252.690 FPS on two nodes. The current rerun preserves
both full-latent hashes and confirms the same performance trend.
