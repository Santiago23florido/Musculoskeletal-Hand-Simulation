"""EMBC 2027 cluster diagnostics and ablations for Pareto-MC on Linux/Mesogip.

This module adds three non-invasive, Slurm-friendly workflows without modifying any
frozen/registered experiment modules or sealed outputs:

1. ``probe``: inspects Linux/HPC resources (GPU, CPU, RAM, disk, Apptainer/Singularity)
   and checks whether the official NeuroMotion container can run on the cluster.
2. ``profile-latency``: measures clean batch latency and per-component breakdown
   (preprocessing+whitening, detection, decision/observer, neural head, SV/SD update)
   for Pareto, L2, and P3 (both CPU head and optional CUDA head) on sub-04 at 20 dB.
3. ``ablate-memory``: zero-retraining ablation on sub-04 comparing frozen P3 seeds 0-2
   with L2 calibration memory enabled (reproducing sealed P3) vs disabled (beta_max=0),
   scored with the original evaluator and paired cluster bootstrap (2000 draws).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import closed_context as cc
from .backbone import replay
from .common import CONDITIONS, OUTPUT, VALIDATION, now, records, sha256, stub, write_new
from .context_c4 import BATCH, load_context
from .data import load_record
from .memory_experiment import SEEDS, read
from .memory_loop import MemoryConfig, make_memory_adapter
from .observer import load_observer
from .posture_c4 import LAG_BATCHES

FOLDER = OUTPUT / "EMBC_cluster"


# --------------------------------------------------------------------------------------
# 1. Cluster & NeuroMotion container probe
# --------------------------------------------------------------------------------------


def probe_environment() -> dict:
    """Collect hardware, software, and container runtime info for cluster execution."""
    commands = {
        "apptainer": ["apptainer", "--version"],
        "singularity": ["singularity", "--version"],
        "docker": ["docker", "--version"],
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    }
    tools = {}
    for name, cmd in commands.items():
        if shutil.which(cmd[0]) is None:
            tools[name] = {"available": False}
            continue
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            tools[name] = {
                "available": res.returncode == 0,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as err:
            tools[name] = {"available": False, "error": str(err)}

    container_ready = any(
        tools[k].get("available", False) for k in ("apptainer", "singularity", "docker")
    )
    disk = shutil.disk_usage(OUTPUT.parent if OUTPUT.parent.exists() else Path.cwd())
    return {
        "probed_at_utc": now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_tmpdir": os.environ.get("SLURM_TMPDIR"),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "tools": tools,
        "neuromotion_container_runtime_available": container_ready,
    }


def run_probe(output_path: Path | None = None) -> dict:
    path = output_path or (FOLDER / "cluster_probe.json")
    info = probe_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    write_new(path, info)
    return info


# --------------------------------------------------------------------------------------
# 2. Component-level latency profiler (CPU and optional CUDA head)
# --------------------------------------------------------------------------------------


class DeviceContextHead(cc.ContextHead):
    """Streaming setrec24 ContextHead executed on a chosen torch device (cpu or cuda)."""

    def __init__(self, model, owner, units, *, device="cpu", keep_logits=False):
        super().__init__(model, owner, units, keep_logits=keep_logits)
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, features, primary):
        owner = self.owner
        event, index, trace = owner.last_event, owner.current_index, owner.state_trace
        if len(trace) != index + 1 or owner.config.batch_size != BATCH:
            raise AssertionError("Head called out of step with the loop")
        x22 = np.asarray(event["features"], dtype=np.float32)
        if x22.shape != tuple(features.shape):
            raise AssertionError("Head features differ from the recorded event")
        q = float(trace[-1][0])
        posture = np.array([q, q - float(trace[max(index - LAG_BATCHES, 0)][0])], np.float32)
        n = len(x22)
        pad = owner.config.ext_fact if index == 0 else 0
        time_arr = np.asarray(event["time"], dtype=np.int64) + index * BATCH + pad
        if n and (time_arr.min() < index * BATCH or time_arr.max() >= (index + 1) * BATCH):
            raise AssertionError("Candidate outside its batch")
        if n:
            x24 = np.concatenate([x22, np.broadcast_to(posture, (n, 2))], 1).astype(np.float32)
            offset = ((time_arr % BATCH) / BATCH).astype(np.float32)
            prim = np.asarray(event["primary"], dtype=bool)
        else:
            x24 = np.zeros((1, 24), np.float32)
            offset = np.zeros(1, np.float32)
            prim = np.asarray([False])
        cells, width = self._cells(np.asarray(event["unit"]), n)
        previous = self.previous
        k = max(width, previous["width"] if previous else 0, 1)
        cells = self._pad(cells, k, -1)
        valid, safe = cells >= 0, cells.clip(0)
        x = x24[safe] * valid[..., None]
        off = offset[safe]
        if previous is None:
            px = np.zeros((self.units, k, 24), np.float32)
            poff = np.zeros((self.units, k), np.float32)
            pvalid = np.zeros((self.units, k), bool)
        else:
            px = self._pad(previous["x"], k)
            poff = self._pad(previous["offset"], k)
            pvalid = self._pad(previous["valid"], k, False)
        t = np.rint(off * BATCH).astype(np.int32)
        pt = np.rint(poff * BATCH).astype(np.int32) - BATCH
        arrays = dict(
            x=x,
            offset=off * valid,
            primary=prim[safe] & valid,
            valid=valid,
            px=px * pvalid[..., None],
            poffset=poff * pvalid,
            time=t,
            keys_time=np.concatenate([pt, t], 1),
            keys_valid=np.concatenate([pvalid, valid], 1),
            posture=np.broadcast_to(posture, (self.units, 2)),
        )
        chunk = {
            name: torch.from_numpy(np.ascontiguousarray(value[:, None])).to(self.device)
            for name, value in arrays.items()
        }
        out, self.hidden = self.model(chunk, self.hidden)
        out = out[:, 0].cpu().numpy()
        logits = np.zeros(n, dtype=np.float32)
        if n:
            logits[cells[valid]] = out[valid]
        self.previous = dict(
            x=x[:, : max(width, 1)],
            offset=off[:, : max(width, 1)],
            valid=valid[:, : max(width, 1)],
            width=width,
        )
        if self.keep_logits:
            self.logits.append(logits.copy())
        return torch.from_numpy(logits)


def attach_component_timers(adapter, *, sync_cuda: bool = False) -> dict[str, list[float]]:
    """Instrument an adapter instance in-place to record per-batch stage timings (ms)."""
    stages: dict[str, list[float]] = {
        "preprocess_ms": [],
        "whiten_ms": [],
        "detect_ms": [],
        "decide_ms": [],
        "head_ms": [],
        "update_sd_ms": [],
        "update_sv_ms": [],
    }

    def _sync():
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _wrap(obj, attr: str, key: str):
        orig = getattr(obj, attr)

        def wrapped(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            _sync()
            stages[key].append((time.perf_counter() - t0) * 1000.0)
            return out

        setattr(obj, attr, wrapped)

    _wrap(adapter, "_preprocess_batch_raw", "preprocess_ms")
    _wrap(adapter, "_whiten", "whiten_ms")
    _wrap(adapter, "_detect_spikes", "detect_ms")
    if (
        getattr(adapter, "head", None) is not None
        or getattr(adapter, "collector", None) is not None
    ):
        _wrap(adapter, "decide", "decide_ms")
    if getattr(adapter, "head", None) is not None:
        orig_head = adapter.head

        def timed_head(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            out = orig_head(*args, **kwargs)
            _sync()
            stages["head_ms"].append((time.perf_counter() - t0) * 1000.0)
            return out

        timed_head.is_identity = orig_head.is_identity
        adapter.head = timed_head
    _wrap(adapter, "_update_spike_det", "update_sd_ms")
    _wrap(adapter, "_update_sep_vectors", "update_sv_ms")
    return stages


def _summarize_ms(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"batches": 0, "median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    return {
        "batches": int(len(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "max_ms": float(np.max(values)),
    }


def make_nomem_settings(settings: dict) -> dict:
    """Return a copy of L2 memory settings with calibration-memory mixing disabled."""
    updated = dict(settings)
    updated["beta_max"] = 0.0
    return updated


def run_latency_profile(*, threads: int = 1, head_device: str = "cpu") -> dict:
    """Profile Pareto, L2, and P3 (CPU or CUDA head) on sub-04 20 dB (5 records)."""
    tag = f"latency_t{threads}_{head_device}"
    out_path = FOLDER / f"{tag}.json"
    if out_path.exists():
        return read(out_path)

    if head_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--head-device cuda requested, but CUDA is not available")

    torch.set_num_threads(threads)
    selected = read(cc.FOLDER / "round3/selected.json")["head"]
    retained = read(OUTPUT / "L2/retained.json")["selection"]
    settings = cc.l2_settings()
    l2_spec = dict(
        kind="l2",
        checkpoint=retained["checkpoint"],
        checkpoint_sha256=retained["checkpoint_sha256"],
        parameters=retained["selection"]["parameters"],
    )

    systems = ["pareto", "l2", "p3_context"]
    if head_device == "cuda":
        systems.append("p3_context_cuda")

    cases = [("sub-04", condition, 20) for condition in CONDITIONS]
    totals: dict[str, list[np.ndarray]] = defaultdict(list)
    components: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in systems}

    for case in cases:
        record = load_record(*case, phase="validation")
        for name in systems:
            if name == "pareto":
                import copy

                from .backbone import build

                adapter = build(record["calibration"], copy.deepcopy(record["config"]))
                sync = False
            elif name == "l2":
                ckpt = OUTPUT / l2_spec["checkpoint"]
                adapter = cc.make_l2_adapter(
                    record, ckpt, settings=settings, **l2_spec["parameters"]
                )
                sync = False
            else:
                ckpt = OUTPUT / selected["checkpoint"]
                adapter = make_memory_adapter(
                    record,
                    None,
                    load_observer(),
                    memory_config=MemoryConfig(**settings),
                    **selected["parameters"],
                )
                adapter.builder = cc.EventRecorder(adapter.builder, adapter)
                dev = "cuda" if name == "p3_context_cuda" else "cpu"
                adapter.head = DeviceContextHead(
                    load_context(ckpt),
                    adapter,
                    record["calibration"].spikes.shape[1],
                    device=dev,
                )
                sync = dev == "cuda"

            stages = attach_component_timers(adapter, sync_cuda=sync)
            pred = replay(record["raw"], adapter, collect_sources=False)
            totals[name].append(pred["batch_ms"])
            for stage_name, vals in stages.items():
                components[name][stage_name].append(np.asarray(vals, dtype=np.float64))
            print(
                f"EMBC profile [{name}] {stub(*case)} "
                f"median={pred['median_batch_ms']:.2f}ms p95={pred['p95_batch_ms']:.2f}ms",
                flush=True,
            )

    summary = {}
    for name in systems:
        summary[name] = {
            "total": _summarize_ms(np.concatenate(totals[name])),
            "components": {
                stage: _summarize_ms(np.concatenate(parts))
                for stage, parts in components[name].items()
            },
        }

    result = {
        "measured_at_utc": now(),
        "threads": threads,
        "head_device": head_device,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "records": [stub(*c) for c in cases],
        "target_ms": 22.0,
        "systems": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_new(out_path, result)
    return result


# --------------------------------------------------------------------------------------
# 3. Zero-retraining P3 memory ablation (sub-04, frozen seeds 0-2)
# --------------------------------------------------------------------------------------


def _ablation_rollout_task(arguments):
    folder, case, spec, disable_memory, threads = arguments
    torch.set_num_threads(threads)
    name = stub(*case)
    meta_path = folder / "metadata" / f"{name}.json"
    if meta_path.exists():
        meta = read(meta_path)
        for key, digest in meta["files"].items():
            if sha256(folder / key) != digest:
                raise ValueError(f"Completed ablation rollout changed: {key}")
        return name

    dest = folder / "predictions" / f"{name}.npz"
    if dest.exists():
        raise ValueError(f"Orphaned prediction in {dest}; clean or use a fresh output dir")

    record = load_record(*case, phase="validation")
    settings = cc.l2_settings()
    if disable_memory:
        settings = make_nomem_settings(settings)
    ckpt = OUTPUT / spec["checkpoint"]
    if sha256(ckpt) != spec["checkpoint_sha256"]:
        raise ValueError("Frozen P3 checkpoint changed")
    adapter = cc.make_context_adapter(
        record, load_context(ckpt), settings=settings, **spec["parameters"]
    )
    prediction = replay(record["raw"], adapter, collect_sources=False)
    prediction["state"] = np.asarray(adapter.state_trace)
    prediction["memory"] = np.asarray(adapter.memory_trace)
    prediction.pop("sources", None)

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **prediction)
    write_new(
        meta_path,
        {
            "predicted_at_utc": now(),
            "record": list(case),
            "spec": spec,
            "disable_memory": disable_memory,
            "threads": threads,
            "labels_opened": False,
            "files": {dest.relative_to(folder).as_posix(): sha256(dest)},
        },
    )
    return name


def run_memory_ablation(*, workers: int = 4, threads: int = 4) -> dict:
    """Compare frozen P3 seeds 0-2 with vs without L2 calibration memory on sub-04."""
    ablation_root = FOLDER / "p3_memory_ablation"
    analysis_path = ablation_root / "analysis.json"
    if analysis_path.exists():
        return read(analysis_path)

    cases = records(VALIDATION)
    heads = cc.final_heads()
    arm_folders: dict[str, Path] = {}

    for variant, disable_memory in (("nomem", True),):
        for seed in SEEDS:
            label = f"p3_{variant}_seed{seed}"
            folder = ablation_root / label
            arm_folders[label] = folder
            tasks = [(folder, case, heads[seed], disable_memory, threads) for case in cases]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_ablation_rollout_task, t) for t in tasks]
                for fut in as_completed(futures):
                    print("EMBC ablation", label, fut.result(), flush=True)
            seal_path = folder / "sealed.json"
            files = {}
            for case in cases:
                meta = read(folder / "metadata" / f"{stub(*case)}.json")
                for key, digest in meta["files"].items():
                    if sha256(folder / key) != digest:
                        raise ValueError("Ablation prediction changed before seal")
                    files[key] = digest
            if not seal_path.exists():
                write_new(
                    seal_path,
                    {
                        "sealed_at_utc": now(),
                        "files": files,
                        "cases": cases,
                        "labels_opened": False,
                    },
                )

    cc.score_folders(list(arm_folders.values()), cases, workers)

    score_dirs = {label: folder / "scores" for label, folder in arm_folders.items()}
    for seed in SEEDS:
        score_dirs[f"p3_full_seed{seed}"] = cc.FOLDER / f"final_validation/seed{seed}/scores"
    score_dirs["pareto"] = OUTPUT / "L0/validation/scores"

    rows = {label: cc.load_rows(folder, cases) for label, folder in score_dirs.items()}
    table = {
        "p3_nomem_vs_p3_full": {
            **cc.mean_of("p3_nomem", 1),
            **cc.mean_of("p3_full", -1),
        },
        "p3_nomem_vs_pareto": {
            **cc.mean_of("p3_nomem", 1),
            "pareto": -1.0,
        },
        "p3_full_vs_pareto": {
            **cc.mean_of("p3_full", 1),
            "pareto": -1.0,
        },
    }
    point_values, effects = cc.bootstrap(rows, cc.clusters_for(cases, "validation"), table)
    result = {
        "analysed_at_utc": now(),
        "scope": "Zero-retraining ablation of L2 calibration memory inside frozen P3 (sub-04)",
        "point_values": point_values,
        "effects": effects,
        "decision_hint": (
            "If p3_nomem_vs_p3_full CI includes 0 (or is > 0), L2 calibration memory is "
            "unnecessary inside P3 and can be dropped for a simpler, cleaner EMBC method."
        ),
    }
    write_new(analysis_path, result)
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="EMBC 2027 cluster experiments for Pareto-MC")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="Inspect cluster hardware, storage, and container runtimes")

    prof = sub.add_parser("profile-latency", help="Component-level latency profile on sub-04 20 dB")
    prof.add_argument("--threads", type=int, default=1)
    prof.add_argument("--head-device", choices=("cpu", "cuda"), default="cpu")

    abl = sub.add_parser("ablate-memory", help="Zero-retraining P3 memory-off ablation on sub-04")
    abl.add_argument("--workers", type=int, default=4)
    abl.add_argument("--threads", type=int, default=4)

    args = parser.parse_args(argv)
    if args.command == "probe":
        print(run_probe(), flush=True)
    elif args.command == "profile-latency":
        res = run_latency_profile(threads=args.threads, head_device=args.head_device)
        print("Latency profile saved:", {k: v["total"] for k, v in res["systems"].items()})
    elif args.command == "ablate-memory":
        res = run_memory_ablation(workers=args.workers, threads=args.threads)
        print("Memory ablation saved:", res["effects"]["p3_nomem_vs_p3_full"]["global_roa_pct"])


if __name__ == "__main__":
    main()
