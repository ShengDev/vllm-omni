# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark layer-partitioned full / serial / stepwise video generation.

An instance runs one complete video warmup, three complete video measurements,
then latent-only correctness requests. Run this program through ``gpu run``.
Only the last measured video is saved; artifact writing is outside timing.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("full", "serial", "stepwise"), required=True)
    parser.add_argument("--gap", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--cond-frames", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditioning", choices=("latent", "latest_kv"), default="latent")
    parser.add_argument(
        "--kv-policy",
        choices=("latest", "clean"),
        default="latest",
        help="KV source policy: latest = replay control (serial+latest replays stepwise "
        "versions); clean = Self Forcing (serial only, predecessors' clean-pass KV)",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="FastVideo/FastWan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--chunk-frames", type=int, default=77)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    if args.chunk_frames < 1 or (args.chunk_frames - 1) % 4:
        parser.error("--chunk-frames must be 1 modulo 4 and at least 5")
    return args


def write_metadata(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    checkout = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(checkout))
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(checkout) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    import numpy as np
    import torch
    from diffusers.utils import export_to_video

    import vllm_omni
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    module_path = Path(vllm_omni.__file__).resolve()
    if not module_path.is_relative_to(checkout):
        raise RuntimeError(f"Expected vllm_omni from {checkout}, imported {module_path}")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"Use a fresh experiment directory; {metadata_path} already exists")

    chunk_frames = args.chunk_frames
    chunk_latent_frames = (chunk_frames - 1) // 4 + 1
    num_frames = args.chunks * chunk_latent_frames * 4 - 3
    if args.kv_policy == "clean" and args.execution != "serial":
        raise ValueError("--kv-policy clean requires --execution serial")
    # Detect the checkpoint family from model_index.json's _class_name — the
    # same signal the pipeline itself uses — instead of matching the model
    # path, so repo IDs and local paths without the family name behave alike.
    model_index_path = Path(args.model) / "model_index.json"
    if not model_index_path.exists():
        raise FileNotFoundError(f"--model must be a local diffusers directory with model_index.json: {args.model}")
    class_name = json.loads(model_index_path.read_text()).get("_class_name", "")
    is_fastwan = class_name == "WanDMDPipeline"
    is_causalwan = class_name == "WanCausalDMDPipeline"
    if not (is_fastwan or is_causalwan):
        raise ValueError(f"Unsupported DMD checkpoint class {class_name!r} in {model_index_path}")
    if is_fastwan:
        dmd_timesteps = (1000.0, 757.0, 522.0)
        scheduler_shift = 8.0
        # FastWan 5B TI2V VAE compresses 16x spatially into 48-channel latents.
        vae_spatial, latent_channels = 16, 48
    else:
        dmd_timesteps = (1000.0, 850.0, 700.0, 550.0, 350.0, 275.0, 200.0, 125.0)
        scheduler_shift = 12.0
        # CausalWan 14B serves 16-channel latents with an 8x-spatial VAE.
        vae_spatial, latent_channels = 8, 16
    num_inference_steps = len(dmd_timesteps)
    mode = "layer" if args.execution == "full" else "chunk"
    schedule = "serial" if args.execution == "full" else args.execution
    model = args.model
    prompt = "A cinematic drone shot over snowy mountains at golden hour"
    samples = (
        json.loads(args.manifest.read_text())["samples"]
        if args.manifest
        else [{"id": "smoke", "prompt": prompt, "seed": 1024}]
    )
    seed = 1024
    sample_id = "smoke"
    run_id = uuid.uuid4().hex
    metadata = {
        "schema_version": 2,
        "status": "initializing",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": str(checkout),
        "checkout_head": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
        "vllm_omni_import": str(module_path),
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": model,
        "model_class_name": class_name,
        "assumed_dmd_timesteps": list(dmd_timesteps),
        "scheduler_shift": scheduler_shift,
        "enable_cpu_offload": args.enable_cpu_offload,
        "kv_source_policy": args.kv_policy if args.conditioning == "latest_kv" else None,
        "prompt": prompt,
        "samples": samples,
        "conditioning": "full_attention" if args.execution == "full" else args.conditioning,
        "quality_frames": "lossless RGB uint8 before video compression, frames.npy",
        "execution": args.execution,
        "gap": 0 if args.execution == "full" else None if args.conditioning == "latest_kv" else args.gap,
        "pipeline_parallel_size": args.pp_size,
        "pipeline_parallel_mode": mode,
        "enforce_eager": True,
        "chunks": args.chunks,
        "chunk_frames": chunk_frames,
        "chunk_latent_frames": chunk_latent_frames,
        "cond_frames": 0 if args.execution == "full" or args.conditioning == "latest_kv" else args.cond_frames,
        "kv_history_chunks": 6 if args.execution != "full" and args.conditioning == "latest_kv" else None,
        "total_latent_frames": args.chunks * chunk_latent_frames,
        "height": 480,
        "width": 832,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "seed": 1024,
        "guidance_scale": 1.0,
        "warmup_iterations": args.warmups,
        "timed_iterations": args.repeats,
        "iterations": [],
        "artifacts": {},
        "timing_scope": {
            "omni_init_wall_ms": "Omni constructor, including automatic engine initialization/warmup",
            "generate_request_wall_ms": "One complete Omni.generate response; excludes artifact saving",
            "performance": "Per-sample median of phase=timed video requests; warmup/repeat counts recorded explicitly",
            "validation": "All latent-only requests excluded from performance statistics",
            "worker_metrics": "CHUNK_PP_METRICS matched by unique experiment_iteration",
        },
    }
    write_metadata(metadata_path, metadata)
    print("LAYER_STEP_CLIENT_CONFIG " + json.dumps(metadata), flush=True)
    omni = None

    def request(phase, index, output_type, request_schedule):
        iteration = f"{run_id}:{sample_id}:{phase}:{index}"
        extra_args = {
            "chunk_frames": chunk_frames,
            "chunk_cond_frames": args.cond_frames,
            "chunk_gap": args.gap,
            "chunk_schedule": request_schedule,
            "collect_pp_metrics": True,
            "experiment_iteration": iteration,
        }
        if args.conditioning == "latest_kv":
            extra_args.update(
                chunk_conditioning="latest_kv",
                kv_history_chunks=6,
                kv_source_policy=args.kv_policy,
            )
        # New params / extra_args per call; no generator or request state reused.
        sampling = OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=1.0,
            output_type=output_type,
            extra_args=extra_args,
        )
        record = {
            "experiment_iteration": iteration,
            "sample_id": sample_id,
            "seed": seed,
            "prompt": prompt,
            "phase": phase,
            "index": index,
            "execution": args.execution,
            "schedule": "full" if args.execution == "full" else request_schedule,
            "gap": 0 if args.execution == "full" else args.gap,
            "output_type": output_type,
            "extra_args": dict(extra_args),
            "status": "running",
        }
        metadata["iterations"].append(record)
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_ITERATION_START " + json.dumps(record), flush=True)
        start = time.perf_counter()
        outputs = omni.generate(
            {"prompt": prompt, "negative_prompt": "", "modalities": ["video"]}, sampling, use_tqdm=False
        )
        record["generate_request_wall_ms"] = (time.perf_counter() - start) * 1000
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], OmniRequestOutput):
            raise TypeError(f"Expected one OmniRequestOutput, received {type(outputs)}")
        result = outputs[0]
        record.update(
            request_id=result.request_id,
            final_output_type=result.final_output_type,
            stage_durations=result.stage_durations,
            worker_peak_memory_mb=result.peak_memory_mb,
        )
        payload = result.latents if output_type == "latent" and result.latents is not None else result.images
        while isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]
        if output_type == "np":
            payload = np.asarray(payload)
            if payload.ndim == 5 and payload.shape[0] == 1:
                payload = payload[0]
            expected = (num_frames, 480, 832, 3)
            if payload.shape != expected:
                raise ValueError(f"Expected video {expected}, received {payload.shape}")
        else:
            if not isinstance(payload, torch.Tensor):
                raise TypeError(f"Expected latent Tensor, received {type(payload)}")
            if payload.ndim == 4:
                payload = payload.unsqueeze(0)
            expected = (
                1,
                latent_channels,
                args.chunks * chunk_latent_frames,
                480 // vae_spatial,
                832 // vae_spatial,
            )
            if tuple(payload.shape) != expected:
                raise ValueError(f"Expected latent {expected}, received {tuple(payload.shape)}")
        record.update(status="complete", output_shape=list(payload.shape), output_dtype=str(payload.dtype))
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_ITERATION_COMPLETE " + json.dumps(record), flush=True)
        return payload, record

    try:
        start = time.perf_counter()
        omni = Omni(
            model=model,
            pipeline_parallel_size=args.pp_size,
            pipeline_parallel_mode=mode,
            enforce_eager=True,
            enable_cpu_offload=args.enable_cpu_offload,
        )
        metadata["omni_init_wall_ms"] = (time.perf_counter() - start) * 1000
        metadata["status"] = "running"
        write_metadata(metadata_path, metadata)
        for sample in samples:
            prompt, seed, sample_id = sample["prompt"], sample["seed"], sample["id"]
            sample_out = args.out / sample_id if args.manifest else args.out
            sample_out.mkdir(exist_ok=True)
            for index in range(metadata["warmup_iterations"]):
                warmup, _ = request("warmup", index, "np", schedule)
                del warmup
            for index in range(metadata["timed_iterations"]):
                frames, record = request("timed", index, "np", schedule)
                if index == metadata["timed_iterations"] - 1:
                    path = sample_out / "video.mp4"
                    rgb = np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)
                    np.save(sample_out / "frames.npy", rgb)
                    # NumPy frames passed to export_to_video must stay in [0, 1];
                    # the helper performs the uint8 scaling itself.
                    export_to_video(list(frames), str(path), fps=16)
                    del rgb
                    metadata["artifacts"][f"{sample_id}:video"] = {
                        "path": str(path.resolve()),
                        "experiment_iteration": record["experiment_iteration"],
                        "fps": 16,
                    }
                    write_metadata(metadata_path, metadata)
                del frames

            validations = [("latent_validation", schedule, "latents.pt", "latents")]
            if args.execution == "stepwise" and (args.gap == 2 or args.conditioning == "latest_kv"):
                validations.append(
                    ("serial_latent_validation", "serial", "latents_serial_validation.pt", "serial_latents")
                )
            for phase, request_schedule, filename, artifact_key in validations:
                latent, record = request(phase, 0, "latent", request_schedule)
                saved = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
                path = sample_out / filename
                torch.save(saved, path)
                metadata["artifacts"][f"{sample_id}:{artifact_key}"] = {
                    "path": str(path.resolve()),
                    "experiment_iteration": record["experiment_iteration"],
                    "source_dtype": str(latent.dtype),
                    "saved_dtype": str(saved.dtype),
                    "shape": list(saved.shape),
                }
                write_metadata(metadata_path, metadata)
                del latent, saved

            stepwise_latent_artifact = metadata["artifacts"].get(f"{sample_id}:latents")
            serial_latent_artifact = metadata["artifacts"].get(f"{sample_id}:serial_latents")
            if stepwise_latent_artifact and serial_latent_artifact:
                stepwise_latents = torch.load(stepwise_latent_artifact["path"], weights_only=True)
                serial_latents = torch.load(serial_latent_artifact["path"], weights_only=True)
                max_abs_diff = float((stepwise_latents - serial_latents).abs().max())
                parity = {"max_abs_diff": max_abs_diff, "identical": max_abs_diff == 0.0}
                metadata.setdefault("validation_conclusions", {})[f"{sample_id}:serial_vs_stepwise"] = parity
                write_metadata(metadata_path, metadata)
                print(f"LAYER_STEP_PARITY {json.dumps({'sample': sample_id, **parity})}", flush=True)
                if not parity["identical"]:
                    raise ValueError(f"serial replay latents diverge from stepwise: max_abs_diff={max_abs_diff}")
                del stepwise_latents, serial_latents
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_metadata(metadata_path, metadata)
        print("LAYER_STEP_CLIENT_COMPLETE " + json.dumps(metadata), flush=True)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_metadata(metadata_path, metadata)
        raise
    finally:
        if omni is not None:
            omni.close()


if __name__ == "__main__":
    main()
