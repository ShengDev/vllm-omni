#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni Latest e2e with long-chunk steady-state FPS.

Runs one (or more) timed Latest request with ``--chunks`` (default 128), then
computes DiT steady FPS on the middle window after dropping ``--drop-head`` /
``--drop-tail`` chunks (default 32 / 32).

Per-chunk DiT finish times come from ``extra_args.chunk_timeline_file`` written
by the final denoise rank.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--deploy-config", type=Path, required=True)
    p.add_argument("--prompt", default="a cat walking on grass")
    p.add_argument("--chunks", type=int, default=128)
    p.add_argument("--denoise-steps", type=int, default=4)
    p.add_argument("--history", type=int, default=6)
    p.add_argument("--warmup-chunks", type=int, default=2, help="throw-away request size before measure")
    p.add_argument("--repeat", type=int, default=1, help="timed full requests (median if >1)")
    p.add_argument("--drop-head", type=int, default=32)
    p.add_argument("--drop-tail", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--stream-decode", action="store_true", default=True)
    p.add_argument("--no-stream-decode", action="store_false", dest="stream_decode")
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--timeline-dir", type=Path, default=None)
    return p.parse_args()


def _frames_for_chunk(chunk: int, n_chunks: int, total_frames: int | None) -> int:
    """Match Omni 7→81 convention when possible: first 9, rest 12."""
    if total_frames == 81 and n_chunks == 7:
        return 9 if chunk == 0 else 12
    if n_chunks <= 1:
        return int(total_frames or 12)
    # Generalize 9 + 12*(N-1) packing when total unknown.
    return 9 if chunk == 0 else 12


def _steady_from_timeline(
    times: dict[int, float],
    *,
    n_chunks: int,
    drop_head: int,
    drop_tail: int,
    total_frames: int | None,
) -> dict:
    if n_chunks <= drop_head + drop_tail:
        raise ValueError(f"chunks={n_chunks} too small for drop_head={drop_head} drop_tail={drop_tail}")
    i0 = drop_head
    i1 = n_chunks - drop_tail  # exclusive end index for window chunks
    missing = [c for c in range(i0, i1) if c not in times]
    if missing:
        raise RuntimeError(f"timeline missing chunks in steady window: {missing[:8]}...")
    t0 = times[i0]
    t1 = times[i1 - 1]
    dt = t1 - t0
    frames = sum(_frames_for_chunk(c, n_chunks, total_frames) for c in range(i0, i1))
    return {
        "window": [i0, i1 - 1],
        "window_chunks": i1 - i0,
        "window_frames": frames,
        "window_s": dt,
        "dit_fps_steady": (frames / dt) if dt > 0 else None,
        "t0": t0,
        "t1": t1,
    }


def _load_timeline(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        chunk_s, ts_s = line.split("\t")
        out[int(chunk_s)] = float(ts_s)
    return out


def _run_once(args: argparse.Namespace, *, chunks: int, timeline_path: Path, label: str) -> dict:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    if timeline_path.exists():
        timeline_path.unlink()

    extra = {
        "num_chunks": chunks,
        "num_denoise_steps": args.denoise_steps,
        "kv_history_chunks": args.history,
        "chunk_schedule": "latest",
        "shift": args.shift,
        "seed": args.seed,
        "stream_decode": bool(args.stream_decode),
        "chunk_timeline_file": str(timeline_path.resolve()),
    }
    print(f"[{label}] chunks={chunks} stream_decode={args.stream_decode} timeline={timeline_path}", flush=True)
    t_req0 = time.perf_counter()
    omni = Omni(model=args.model, deploy_config=str(args.deploy_config))
    try:
        outs = omni.generate(
            args.prompt,
            OmniDiffusionSamplingParams(extra_args=extra),
        )
    finally:
        close = getattr(omni, "close", None) or getattr(omni, "shutdown", None)
        if callable(close):
            close()
    t_req1 = time.perf_counter()
    n_frames = None
    if outs:
        images = getattr(outs[0], "images", None) or []
        if images:
            video = images[0]
            import torch

            if isinstance(video, dict):
                video = video.get("video") or video.get("frames")
            if isinstance(video, torch.Tensor):
                t = video
                if t.ndim == 5:
                    t = t[0]
                # T,H,W,C or C,T,H,W
                if t.ndim == 4 and t.shape[-1] in (3, 4):
                    n_frames = int(t.shape[0])
                elif t.ndim == 4 and t.shape[0] in (3, 4):
                    n_frames = int(t.shape[1])
                elif t.ndim == 4:
                    n_frames = int(t.shape[0])
    times = _load_timeline(timeline_path) if timeline_path.is_file() else {}
    steady = None
    if times:
        steady = _steady_from_timeline(
            times,
            n_chunks=chunks,
            drop_head=args.drop_head,
            drop_tail=args.drop_tail,
            total_frames=n_frames,
        )
    e2e_s = t_req1 - t_req0
    total_frames = n_frames
    if total_frames is None and times:
        total_frames = sum(_frames_for_chunk(c, chunks, None) for c in range(chunks))
    return {
        "label": label,
        "chunk_schedule": "latest",
        "chunks": chunks,
        "frames": total_frames,
        "request_s": e2e_s,
        "e2e_fps": (total_frames / e2e_s) if total_frames and e2e_s > 0 else None,
        "timeline_chunks": len(times),
        "steady": steady,
        "timeline_file": str(timeline_path),
    }


def main() -> None:
    args = _parse_args()
    out_dir = args.json_out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tl_dir = args.timeline_dir or (out_dir / "timelines")
    tl_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup_chunks > 0:
        warm_tl = tl_dir / "warmup.tsv"
        try:
            _run_once(args, chunks=args.warmup_chunks, timeline_path=warm_tl, label="warmup")
        except Exception as exc:  # noqa: BLE001 — keep measure even if warmup fails to close cleanly
            print(f"[warmup] failed (continuing): {exc}", flush=True)

    runs = []
    for i in range(args.repeat):
        tl = tl_dir / f"measure_{i}.tsv"
        runs.append(_run_once(args, chunks=args.chunks, timeline_path=tl, label=f"measure[{i}]"))

    steady_fps = [r["steady"]["dit_fps_steady"] for r in runs if r.get("steady") and r["steady"].get("dit_fps_steady")]
    e2e_fps = [r["e2e_fps"] for r in runs if r.get("e2e_fps")]
    report = {
        "engine": "omni",
        "scheme": "Omni-VL",
        "chunk_schedule": "latest",
        "chunks": args.chunks,
        "drop_head": args.drop_head,
        "drop_tail": args.drop_tail,
        "denoise_steps": args.denoise_steps,
        "history": args.history,
        "warmup_chunks": args.warmup_chunks,
        "repeat": args.repeat,
        "aggregation": "median over timed requests",
        "dit_fps_steady_median": statistics.median(steady_fps) if steady_fps else None,
        "e2e_fps_median": statistics.median(e2e_fps) if e2e_fps else None,
        "runs": runs,
    }
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"WROTE {args.json_out} dit_fps_steady_median={report['dit_fps_steady_median']} "
        f"e2e_fps_median={report['e2e_fps_median']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
