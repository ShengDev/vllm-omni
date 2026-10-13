# Wan chunk pipeline: IPC/NCCL benchmark adapters

This opt-in harness runs the native `StageWanTransformer`, chunk executor,
Latest KV selection, RF Flow Euler sampler and paged FlashAttention from
[PR #8282](https://github.com/vllm-project/vllm-omni/pull/8282).
It publishes KV per transformer block using CUDA IPC on the same host and
NCCL between hosts. Native attention, model weights and sampling stay in use.
The fused variant also caches request-local conditioning projections and
preserves eager BF16 rounding in modulation and gated residual kernels.

These are benchmark-scoped adapters, installed and restored by context
managers. The diffusion frontend does not gain a multinode serving executor.
Each process handles one request at a time; this harness does not measure
concurrent serving or VAE/end-to-end video throughput.

## Reproduce

Use a Linux/CUDA environment compatible with the PR #8282 branch, with vLLM,
PyTorch, paged FlashAttention, Triton, `safetensors` and `cuda.bindings` installed.
The tested environment and raw measurements are documented in
[the results](results/h800-q3-20261009/README.md).

`OMNI_MODEL` must point to the complete Wan 2.1 1.3B RF Diffusers model root,
including `transformer/config.json` and transformer safetensors. Passing only
the transformer directory is unsupported. The model configuration must have
30 blocks, 12 attention heads and head size 128.

The compressed fixture contains BF16 T5 embeddings for `a cat walking on grass`,
shape `[1,512,4096]`. It excludes T5 loading and encoding from DiT measurements.
Expand it to an absolute path visible from both nodes:

```bash
gzip -dc benchmarks/ar_diffusion/results/h800-q3-20261009/conditioning.pt.gz > /tmp/omni-condition.pt
sha256sum /tmp/omni-condition.pt
# 198358abb9eb6e80296d18bb78f82924a4fef1fa1a44ff26af34e75cfda9c5af
```

For two nodes, set `OMNI_NODE_RANK` to `0` or `1`, `OMNI_MASTER_ADDR` to the
first node's reachable address, `OMNI_CONDITION` to the expanded fixture path,
and `OMNI_OUTPUT` to an empty shared results directory. Run on both nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 \
python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=5 --node-rank="$OMNI_NODE_RANK" \
  --master-addr="$OMNI_MASTER_ADDR" --master-port=29879 \
  -m benchmarks.ar_diffusion.run_multinode_wan \
  --model "$OMNI_MODEL" --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 2 --steps 4 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3 \
  --expected-sha a9a53546cb02479643554cce66ee217c7c00d84af6cbb16a0ecde54b60c2f45d
```

Adjust the network interface to the actual deployment. Each node uses five
GPUs; the stage-major placement has five stages with two block partitions per
stage (`K=15`). CUDA IPC requires distinct physical GPUs for local ranks.

For one node (`K=30`), use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m torch.distributed.run --standalone --nproc-per-node=5 \
  -m benchmarks.ar_diffusion.run_multinode_wan \
  --model "$OMNI_MODEL" --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 1 --steps 4 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3 \
  --expected-sha d37a9c999e612e6578aad0ff8fd715ba3d5abfa4a46994f698adb3a8df9396fe
```

Run from the checkout root and use a distinct output directory for each
topology or source revision. `hybrid` enables the transport and conditioning
cache without pointwise fusion; `cached` enables only the conditioning cache.
Only the four-denoise-step plus clean contract is accepted by this harness.

## Timing and output contract

Each request has 128 chunks of three latent frames at BF16 shape
`[1,16,3,60,104]`, seed zero, history six chunks and no sink. The complete latent
is `[1,16,384,60,104]`; decoding it would produce 1533 frames at 832x480.

Steady FPS is a pixel-frame equivalent of pure DiT throughput. The final
denoise rank records one CUDA completion event per chunk. After 64 skipped
chunks, 32 measured chunks contribute 384 frame equivalents:

```text
FPS = 384000 / (completion_ms[95] - completion_ms[63])
```

The remaining 32 chunks complete the request. Text encoding, model loading,
VAE, hashing and video encoding are excluded. A full request warms each
variant, followed by three full measured requests; the report uses medians.
The harness retains all latents, checks finiteness and hashes the complete
output after timing. All variants and repeats must match the supplied hash
or the first request's hash. Latest KV semantics depend on `K`, so hashes are
compared within a topology rather than across `K=15` and `K=30`.

The hybrid ring has `(history+1) * stages = 35` version positions per layer;
it owns the only KV pool in hybrid runs. The baseline allocates its native
version pool separately. `kv_capacity` reports version positions per layer
for the selected pool, identified by `kv_pool`; `kv_reserved_bytes` includes
the hybrid ticket storage when applicable.
Buffers use producer-ready and consumer-release tickets to prevent a reused
ring position from overwriting KV that attention still reads. Failed IPC
requests keep mappings alive until the worker job exits.

Source snapshots, per-request events and hashes, GPU memory statistics and
median summaries are written under the output directory. Per-block transport
rounds and read labels derive directly from the native `ChunkPlan`; no second
scheduler or general KV manager is included. Retained WaveServe transport
source attribution and adaptation hashes are listed in
[provenance.json](provenance.json).

## Tests

CPU tests cover native KV labels, last-reader release identities, physical
pool addresses, allocation boundaries, exported-buffer lifetime on failure,
cache expiration and restoration of modules after normal and failed requests.
CUDA tests compare fused kernels against separate eager BF16 operations,
including strided residual tensors. They do not require model weights:

```bash
python -m pytest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_hybrid_kv.py \
  tests/diffusion/ar_diffusion/test_wan_native_optimizations.py \
  -m 'core_model and cpu' --run-level=L1 -q
python -m pytest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_native_pointwise.py \
  -m 'core_model and cuda' --run-level=L2 -q
```

The repository-wide test plugins also require a vLLM version compatible with
the selected Omni branch. For isolated unit/kernel checks when those unrelated
plugins cannot load, add `--noconftest` and omit `--run-level`.
