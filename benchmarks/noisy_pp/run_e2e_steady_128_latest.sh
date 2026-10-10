#!/bin/bash
# Omni Latest + WaveServe VL, 128 chunks, steady FPS on middle (drop head/tail 32).
# No serial. Aligns with Desktop README workload (S=5/T=4/history=6/shift=5) but longer.
set -euo pipefail

PROMPT="${PROMPT:-a cat walking on grass}"
CHUNKS="${CHUNKS:-128}"
DROP_HEAD="${DROP_HEAD:-32}"
DROP_TAIL="${DROP_TAIL:-32}"
DENOISE="${DENOISE:-4}"
HISTORY="${HISTORY:-6}"
WARMUP="${WARMUP:-2}"
REPEAT="${REPEAT:-1}"

PY="${PY:-/data/sheng/envs/omni-e2e-v031/bin/python}"
[ -x "$PY" ] || PY=/data/dxw/env/kernel-pr-v031-20261007/bin/python
MODEL="${MODEL:-/data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev}"
REPO="${OMNI_REPO:-/data/sheng/vllm-omni-exp-chunk-pp}"
WS="${WS_REPO:-/data/sheng/WaveServe-dev}"
OUT="${OUT:-/data/sheng/outputs/e2e_steady_128_latest}"
LOG="${LOG:-/data/sheng/logs/e2e_steady_128_latest.log}"

mkdir -p "$OUT" "$(dirname "$LOG")"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "===== host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(date -Is) ====="
echo "Omni HEAD: $(cd "$REPO" && git log -1 --oneline || true)"
echo "WS HEAD:   $(cd "$WS" && git log -1 --oneline 2>/dev/null || echo missing)"
echo "PY=$PY chunks=$CHUNKS drop=$DROP_HEAD/$DROP_TAIL warmup=$WARMUP repeat=$REPEAT"
nvidia-smi -L || true

export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1
cd "$REPO"

echo
echo "########## 1) Omni deploy S=5 T=$DENOISE (latest only) ##########"
"$PY" - <<PY
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load("$REPO/vllm_omni/deploy/waveserve_wan.yaml")
s0 = cfg.stages[0]
OmegaConf.update(s0, "parallel_config.pipeline_parallel_size", 5, force_add=True)
OmegaConf.update(s0, "model_config.ar_diffusion_stage_config.stage_parallel_size", 5, force_add=True)
OmegaConf.update(s0, "model_config.ar_diffusion_stage_config.max_history_chunks", $HISTORY, force_add=True)
OmegaConf.update(s0, "default_sampling_params.extra_args.num_denoise_steps", $DENOISE, force_add=True)
OmegaConf.update(s0, "default_sampling_params.extra_args.num_chunks", $CHUNKS, force_add=True)
OmegaConf.update(s0, "default_sampling_params.extra_args.chunk_schedule", "latest", force_add=True)
out = Path("$OUT/omni_s5_deploy.yaml")
out.write_text(OmegaConf.to_yaml(cfg))
print("wrote", out)
PY

echo
echo "########## 2) Omni Latest steady ($CHUNKS chunks) ##########"
"$PY" benchmarks/noisy_pp/e2e_steady_128_latest.py \
  --model "$MODEL" \
  --deploy-config "$OUT/omni_s5_deploy.yaml" \
  --prompt "$PROMPT" \
  --chunks "$CHUNKS" \
  --denoise-steps "$DENOISE" \
  --history "$HISTORY" \
  --warmup-chunks "$WARMUP" \
  --repeat "$REPEAT" \
  --drop-head "$DROP_HEAD" \
  --drop-tail "$DROP_TAIL" \
  --stream-decode \
  --json-out "$OUT/omni_vl_c${CHUNKS}_steady.json" \
  --timeline-dir "$OUT/timelines_omni"

echo
echo "########## 3) WaveServe-dev VL ($CHUNKS chunks) ##########"
if [[ -d "$WS" && -f "$WS/bench.py" ]]; then
  export PYTHONPATH="$WS"
  cd "$WS"
  # Use 6 procs as prior e2e; VL vertical/latest.
  "$PY" -m torch.distributed.run --standalone --nproc-per-node=6 bench.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --shift 5.0 \
    --layout vertical --kv latest --blocks-per-rank 30 \
    --steps "$DENOISE" --history "$HISTORY" --chunks "$CHUNKS" \
    --warmup "$WARMUP" --seed 0 \
    --device cuda --dtype bfloat16 \
    --report "$OUT/waveserve_vl_${CHUNKS}.json" \
    --video "$OUT/waveserve_vl_${CHUNKS}.mp4" || echo "WS_FAILED continue to summarize"

  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$OUT/waveserve_vl_${CHUNKS}.json")
if not p.is_file():
    print("no WS report")
    raise SystemExit(0)
d = json.loads(p.read_text())
tl = d.get("timeline") or []
n = int(d.get("chunks") or len(tl))
drop_h, drop_t = $DROP_HEAD, $DROP_TAIL
i0, i1 = drop_h, n - drop_t
if len(tl) < n or i1 <= i0:
    print("WS timeline too short", len(tl), n)
    raise SystemExit(0)
# dit_finished is relative to start; absolute deltas from first/last in window
frames = [int(r["frames"]) for r in tl]
t_dit = [float(r["dit_finished"]) for r in tl]
win_frames = sum(frames[i0:i1])
dt = t_dit[i1 - 1] - t_dit[i0]
steady = win_frames / dt if dt > 0 else None
out = {
    "engine": "waveserve",
    "scheme": d.get("scheme"),
    "chunks": n,
    "drop_head": drop_h,
    "drop_tail": drop_t,
    "frames": d.get("frames"),
    "dit_fps_readme": d.get("dit_fps"),  # upstream 1..N-1 definition
    "e2e_fps": d.get("e2e_fps"),
    "first_frame_s": d.get("first_frame_s"),
    "steady": {
        "window": [i0, i1 - 1],
        "window_chunks": i1 - i0,
        "window_frames": win_frames,
        "window_s": dt,
        "dit_fps_steady": steady,
    },
}
Path("$OUT/waveserve_vl_${CHUNKS}_steady.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"WROTE WS steady dit_fps_steady={steady} (window chunks {i0}..{i1-1})")
PY
else
  echo "WS repo missing; skip"
fi

echo
echo "########## 4) REPORT ##########"
"$PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT")
omni_p = out / "omni_vl_c${CHUNKS}_steady.json"
ws_p = out / "waveserve_vl_${CHUNKS}_steady.json"
omni = json.loads(omni_p.read_text()) if omni_p.is_file() else {}
ws = json.loads(ws_p.read_text()) if ws_p.is_file() else {}
md = []
md.append("# Steady-state Latest e2e (128 chunks)")
md.append("")
md.append(f"- Workload: T={$DENOISE}, history={$HISTORY}, chunks={$CHUNKS}, schedule=**latest only**")
mid = int("$CHUNKS") - int("$DROP_HEAD") - int("$DROP_TAIL")
md.append(f"- Steady window: drop head={$DROP_HEAD}, drop tail={$DROP_TAIL} → middle {mid} chunks")
md.append(f"- Warmup chunks={$WARMUP}, measure repeat={$REPEAT}")
md.append("- dit_fps_steady = window_pixel_frames / (dit_finish[last_mid] - dit_finish[first_mid])")
md.append("")
md.append("| Scheme | dit_fps_steady | e2e_fps | notes |")
md.append("|---|---:|---:|---|")
def fmt(x):
    return "n/a" if x is None else f"{float(x):.3f}"
o_steady = omni.get("dit_fps_steady_median")
o_e2e = omni.get("e2e_fps_median")
md.append(f"| Omni-VL | {fmt(o_steady)} | {fmt(o_e2e)} | cell-level Noisy PP |")
w_steady = (ws.get("steady") or {}).get("dit_fps_steady")
md.append(f"| WaveServe-VL | {fmt(w_steady)} | {fmt(ws.get('e2e_fps'))} | upstream timeline |")
(out / "REPORT.md").write_text("\n".join(md) + "\n")
cmp = {"omni": omni, "waveserve": ws, "chunks": $CHUNKS, "drop_head": $DROP_HEAD, "drop_tail": $DROP_TAIL}
(out / "comparison.json").write_text(json.dumps(cmp, indent=2) + "\n")
print((out / "REPORT.md").read_text())
print("WROTE", out / "REPORT.md", out / "comparison.json")
PY

date -Is > "$OUT/completed.txt"
echo DONE
