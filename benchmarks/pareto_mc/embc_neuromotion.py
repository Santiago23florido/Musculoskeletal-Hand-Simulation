"""Phase 2 (EMBC 2027): Confirmatory unseen subjects and OOD movements via MUniverse NeuroMotion.

Generates brand-new virtual subjects using the preregistered seeds from phase0.py
([210601, 210602, 210603]) via the official Singularity/Apptainer NeuroMotion container
(docker://pranavm19/muniverse:neuromotion), calibrates the first 5 s with the exact FDSI
CBSS recipe, and evaluates frozen Pareto, L2 (3 seeds), P3 (3 seeds), and P3-nomem
(3 seeds) with zero retraining and zero threshold tuning.

Conditions (90 s each, 5 s isometric calibration at 0 deg + 80 s movement + 5 s rest):
  - Matched (in-distribution): triangular-ramp10s (4 deg/s), triangular-ramp5s (8 deg/s)
  - Out-of-distribution (OOD): triangular-ramp3s (13.3 deg/s), sinusoid-0.1Hz (smooth 0-40 deg)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import closed_context as cc
from .backbone import build, replay
from .common import OUTPUT, ROOT, now, reference, sha256, write_new
from .context_c4 import load_context
from .embc_cluster import make_nomem_settings
from .evaluate import ramp_cycles
from .memory_experiment import SEEDS, read

FOLDER = OUTPUT / "EMBC_neuromotion"
FS = 2048
DURATION_S = 90.0
CALIB_S = 5.0
EFFORT_LEVEL = 0.15
PEAK_ANGLE_DEG = 40.0
BOOTSTRAP = 2000

CONFIRMATORY_SUBJECTS = (
    ("conf-01", 210601),
    ("conf-02", 210602),
    ("conf-03", 210603),
)
CONDITIONS = (
    "triangular-ramp10s",
    "triangular-ramp5s",
    "triangular-ramp3s",
    "sinusoid-0.1Hz",
)
MATCHED_CONDITIONS = ("triangular-ramp10s", "triangular-ramp5s")
OOD_CONDITIONS = ("triangular-ramp3s", "sinusoid-0.1Hz")
SNRS = (30, 20)


# --------------------------------------------------------------------------------------
# Movement profile generators (exact 90 s: 5 s calib at 0 deg + 80 s movement + 5 s rest)
# --------------------------------------------------------------------------------------


def make_profiles(condition: str, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Return (effort_profile, angle_profile) of length 90 * fs (184,320 samples)."""
    total = int(round(DURATION_S * fs))
    calib = int(round(CALIB_S * fs))
    active_s = DURATION_S - 2.0 * CALIB_S  # 80.0 s
    active_n = int(round(active_s * fs))

    effort = np.full(total, EFFORT_LEVEL, dtype=np.float32)
    angle = np.zeros(total, dtype=np.float32)

    if condition.startswith("triangular-ramp") and condition.endswith("s"):
        ramp_s = float(condition.removeprefix("triangular-ramp").removesuffix("s"))
        half = int(round(ramp_s * fs))
        up = np.linspace(0.0, PEAK_ANGLE_DEG, half, endpoint=False, dtype=np.float32)
        down = np.linspace(PEAK_ANGLE_DEG, 0.0, half, endpoint=False, dtype=np.float32)
        cycle = np.concatenate([up, down])
        reps = active_n // len(cycle)
        seg = np.tile(cycle, reps)
        angle[calib : calib + len(seg)] = seg
    elif condition == "sinusoid-0.1Hz":
        t = np.arange(active_n, dtype=np.float64) / fs
        wave = 0.5 * PEAK_ANGLE_DEG * (1.0 - np.cos(2.0 * np.pi * 0.1 * t))
        angle[calib : calib + active_n] = wave.astype(np.float32)
    else:
        raise ValueError(f"Unsupported NeuroMotion condition: {condition}")

    return effort, angle


def make_neuromotion_config(subject_seed: int, *, fibre_density: int = 185) -> dict:
    """Build an FDSI Flexion-Extension NeuroMotion configuration matching the benchmark."""
    return {
        "PathToBioMimeWeights": "./ckp/model_linear.pth",
        "MorphMUAPS": True,
        "SubjectConfiguration": {
            "SubjectSeed": int(subject_seed),
            "FibreDensity": int(fibre_density),
            "MuscleLabels": ["ECRB", "ECRL", "ECU", "EDI", "PL", "FCU", "FDSI"],
            "MuscleMotorUnitCounts": [186, 204, 180, 186, 164, 422, 120],
        },
        "MovementConfiguration": {
            "TargetMuscle": "FDSI",
            "MovementType": "Dynamic",
            "MovementDOF": "Flexion-Extension",
            "MovementProfileParameters": {
                "MovementDuration": DURATION_S,
                "TargetEffort": int(round(EFFORT_LEVEL * 100)),
            },
        },
        "RecordingConfiguration": {
            "SamplingFrequency": FS,
            "ElectrodeConfiguration": {
                "InterElectrodeDistance": 8,
                "NElectrodes": 320,
                "NRows": 10,
                "NCols": 32,
                "DesiredNCols": 10,
            },
            "FilterProperties": {
                "FilterType": "ButterWorth",
                "CutoffFrequency": 800,
                "FilterOrder": 4,
            },
            "NoiseSeed": None,
            "NoiseLeveldb": None,
        },
    }


def select_optimal_10x10(emg_3d: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Select the 10 highest-energy contiguous wrapped columns -> (T, 100) float32."""
    if emg_3d.ndim != 3 or emg_3d.shape[0] != 10:
        raise ValueError(f"Expected (10, NCols, T) EMG array, got {emg_3d.shape}")
    n_cols = emg_3d.shape[1]
    desired = 10
    rms_per_col = np.sqrt(np.mean(np.square(emg_3d, dtype=np.float64), axis=(0, 2)))
    center = int(np.argmax(rms_per_col))
    half = desired // 2
    cols = [(center - half + i) % n_cols for i in range(desired)]
    sub = emg_3d[:, cols, :]  # (10, 10, T)
    return np.ascontiguousarray(sub.reshape(100, -1).T, dtype=np.float32), cols


def spikes_list_to_binary(spikes_obj, n_samples: int) -> np.ndarray:
    """Convert NeuroMotion list-of-firing-indices into a (T, M) uint8 binary matrix."""
    n_units = len(spikes_obj)
    binary = np.zeros((n_samples, n_units), dtype=np.uint8)
    for u, firings in enumerate(spikes_obj):
        idx = np.asarray(firings, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        if len(idx):
            binary[idx, u] = 1
    return binary


def add_gaussian_noise(clean_emg: np.ndarray, snr_db: int, seed: int) -> np.ndarray:
    """Add deterministic white Gaussian noise at the requested SNR (dB)."""
    rng = np.random.RandomState(seed)
    std_noise = float(np.std(clean_emg, dtype=np.float64)) * (10.0 ** (-snr_db / 20.0))
    return (clean_emg + rng.normal(0.0, std_noise, clean_emg.shape)).astype(np.float32)


def record_stub(subject: str, condition: str, snr: int) -> str:
    return f"{subject}_FDSI_{condition}_snr{snr}dB"


def fdsi_adapt_config():
    """Return the exact pinned FDSI Pareto AdaptConfig (fixed-step, spike-based)."""
    reference()
    from adapt_decomp.adaptation import AdaptConfig

    return AdaptConfig(
        adapt_sd=True,
        adapt_sv=True,
        adapt_wh=True,
        batch_ms=100,
        centroid_momentum=0.95,
        ch_map=None,
        ch_mask=None,
        compute_loss=False,
        contrast_fun="logcosh",
        contrast_scope="spike_based",
        cov_alpha=0.1,
        debug=False,
        device="cpu",
        ema_alpha=0.95,
        eps=1e-7,
        ext_fact=10,
        ext_mode="block",
        fifo_length=None,
        filter_order=4,
        fs=FS,
        highcut=500.0,
        lowcut=20.0,
        lr_mode="fixed",
        max_sigma_batches=1,
        notch_n_harmonics=3,
        notch_order=2,
        notch_width_hz=1.0,
        powerline=True,
        powerline_freq=50.0,
        replace_bad_channels=False,
        safety_clip_multiplier_sv=20.0,
        safety_clip_multiplier_wh=20.0,
        save_params=False,
        shrinkage=0.001,
        source_fifo_batches=2,
        spike_det_exp=2.0,
        spike_height_mult=3,
        spike_min_dist_ms=10,
        spike_prev_weight=5,
        sv_epochs=1,
        sv_learning_rate=0.0024222252021885837,
        sv_tol=0.0001,
        wh_learning_rate=0.04038249056090869,
        wh_mode="kl_to_identity",
        wh_sv_coupling=True,
    )


def fdsi_cbss_config(random_seed: int = 42):
    """Return the exact FDSI CBSSConfig used for 5-s isometric calibration."""
    reference()
    from adapt_decomp import CBSSConfig

    return CBSSConfig(
        fs=FS,
        preprocess_emg=True,
        lowcut=20.0,
        highcut=500.0,
        filter_order=4,
        powerline=True,
        powerline_freq=50.0,
        notch_width_hz=1.0,
        notch_n_harmonics=3,
        notch_order=2,
        replace_bad_channels=False,
        ext_fact=10,
        ext_mode="block",
        n_components=None,
        whitening_method="ZCA",
        regularization="auto",
        eps=1e-10,
        contrast_fun="square",
        contrast_exp=3.0,
        search_iter=300,
        ica_iter=300,
        ica_tol=1e-4,
        spike_det_exp=2.0,
        spike_min_dist_ms=10.0,
        refinement_loop=True,
        refinement_mode="sil",
        refine_max_iter=20,
        sil_th=0.9,
        min_spikes=10,
        roa_th=0.3,
        run_duplicate_removal=True,
        compute_properties=True,
        save_emg=True,
        device="cpu",
        dtype="float32",
        random_seed=random_seed,
        verbose=False,
    )


# --------------------------------------------------------------------------------------
# Step 1: Containerized NeuroMotion generation (reuses per-subject MUAP library)
# --------------------------------------------------------------------------------------


def _detect_container_engine() -> str:
    for binary in ("singularity", "apptainer"):
        if shutil.which(binary):
            return binary
    raise RuntimeError("Neither singularity nor apptainer found in PATH")


def _run_container(
    engine: str,
    sif_path: Path,
    script_path: Path,
    run_dir: Path,
    *,
    use_gpu: bool = True,
):
    cmd = [engine, "exec"]
    if use_gpu:
        cmd.append("--nv")
    cmd += [
        "--cleanenv",
        "-B",
        f"{script_path.resolve()}:/opt/NeuroMotion/run_neuromotion.py",
        "-B",
        f"{run_dir.resolve()}:/run_dir",
        str(sif_path.resolve()),
        "bash",
        "-c",
        (
            "source /opt/mambaforge/etc/profile.d/conda.sh && "
            "conda activate NeuroMotion && "
            "cd /opt/NeuroMotion && "
            "python run_neuromotion.py --run_dir /run_dir"
        ),
    ]
    subprocess.run(cmd, check=True)


def generate_dataset(sif_path: Path, *, subjects=CONFIRMATORY_SUBJECTS, conditions=CONDITIONS):
    """Generate clean + noisy confirmatory recordings via NeuroMotion Singularity container."""
    engine = _detect_container_engine()
    script_path = ROOT / "external/pareto_mc_muniverse/src/muniverse/datasets/_run_neuromotion.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing MUniverse runner script: {script_path}. "
            "Clone https://github.com/dfarinagroup/muniverse into external/pareto_mc_muniverse."
        )
    if not sif_path.exists():
        raise FileNotFoundError(f"Container image not found: {sif_path}")

    data_root = FOLDER / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    use_gpu = torch.cuda.is_available()

    for subject, seed in subjects:
        subj_dir = data_root / subject
        clean_dir = subj_dir / "clean"
        noisy_dir = subj_dir / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        # Keep MUAP library temporarily in /tmp (or local scratch) so NFS never stores gigabytes
        muaps_cache: dict[str, np.ndarray] = {}
        selected_cols_subject: list[int] | None = None

        for cond_idx, condition in enumerate(conditions):
            base = f"{subject}_FDSI_{condition}"
            emg_clean_path = clean_dir / f"{base}_emg.npz"
            spikes_path = clean_dir / f"{base}_spikes.npz"
            angle_path = clean_dir / f"{base}_angle.npz"
            meta_path = clean_dir / f"{base}_metadata.json"

            all_noisy_exist = all(
                (noisy_dir / f"{record_stub(subject, condition, snr)}_emg.npz").exists()
                for snr in SNRS
            )
            if (
                emg_clean_path.exists()
                and spikes_path.exists()
                and angle_path.exists()
                and all_noisy_exist
            ):
                print(f"[SKIP] {base} already generated", flush=True)
                continue

            effort, angle = make_profiles(condition)
            # Vary spike RNG per condition while keeping same subject MUAP pool
            cfg = make_neuromotion_config(seed + cond_idx * 1000)

            with tempfile.TemporaryDirectory(prefix=f"nm_{subject}_{condition}_") as tmp:
                run_dir = Path(tmp)
                (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                inputs = {"effort_profile": effort, "angle_profile": angle}
                if muaps_cache:
                    inputs["muaps"] = muaps_cache["muaps"]
                    inputs["muap_angle_labels"] = muaps_cache["muap_angle_labels"]
                np.savez(run_dir / "input_data.npz", **inputs)

                print(
                    f"[GEN] {base} (seed={seed}, reuse_muaps={bool(muaps_cache)}, gpu={use_gpu})",
                    flush=True,
                )
                _run_container(engine, sif_path, script_path, run_dir, use_gpu=use_gpu)

                out_npz = run_dir / "emg_data.npz"
                with np.load(out_npz, allow_pickle=True) as arc:
                    emg_3d = np.asarray(arc["emg"], dtype=np.float32)
                    spikes_obj = arc["spikes"]
                    if not muaps_cache:
                        muaps_cache["muaps"] = np.asarray(arc["muaps"], dtype=np.float32)
                        muaps_cache["muap_angle_labels"] = np.asarray(arc["muap_angle_labels"])

            if selected_cols_subject is None:
                clean_emg, selected_cols_subject = select_optimal_10x10(emg_3d)
            else:
                sub = emg_3d[:, selected_cols_subject, :]
                clean_emg = np.ascontiguousarray(sub.reshape(100, -1).T, dtype=np.float32)

            binary_spikes = spikes_list_to_binary(spikes_obj, len(angle))
            np.savez_compressed(emg_clean_path, emg=clean_emg)
            np.savez_compressed(spikes_path, spikes=binary_spikes)
            np.savez_compressed(angle_path, angle=angle)

            for snr in SNRS:
                noise_seed = seed * 10 + cond_idx * 100 + snr
                noisy_emg = add_gaussian_noise(clean_emg, snr, noise_seed)
                np.savez_compressed(
                    noisy_dir / f"{record_stub(subject, condition, snr)}_emg.npz",
                    emg=noisy_emg,
                )

            if not meta_path.exists():
                write_new(
                    meta_path,
                    {
                        "generated_at_utc": now(),
                        "subject": subject,
                        "subject_seed": seed,
                        "condition": condition,
                        "fs": FS,
                        "duration_s": DURATION_S,
                        "selected_columns": selected_cols_subject,
                        "n_units_total": int(binary_spikes.shape[1]),
                        "n_active_units": int((binary_spikes.sum(0) > 0).sum()),
                        "container": str(sif_path.name),
                        "engine": engine,
                    },
                )
            active_count = int((binary_spikes.sum(0) > 0).sum())
            print(f"[DONE] {base}: emg={clean_emg.shape}, active_units={active_count}", flush=True)

        # Explicitly free subject MUAP cache before next subject
        muaps_cache.clear()


# --------------------------------------------------------------------------------------
# Step 2: 5-second CBSS calibration (matching exact FDSI protocol)
# --------------------------------------------------------------------------------------


def _calibrate_one(args):
    subject, condition, snr = args
    torch.set_num_threads(2)
    reference()
    from adapt_decomp import CBSS

    name = record_stub(subject, condition, snr)
    cal_dir = FOLDER / "calibration" / subject
    cal_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = cal_dir / f"{name}_cbss.pkl"
    meta_path = cal_dir / f"{name}_cbss.json"
    if pkl_path.exists() and meta_path.exists():
        return name, read(meta_path)["n_units_kept"]

    noisy_path = FOLDER / "data" / subject / "noisy" / f"{name}_emg.npz"
    spikes_path = FOLDER / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz"
    with np.load(noisy_path, allow_pickle=False) as arc:
        emg = np.asarray(arc["emg"], dtype=np.float32)
    with np.load(spikes_path, allow_pickle=False) as arc:
        gt_binary = np.asarray(arc["spikes"], dtype=np.uint8)

    n_cal = int(CALIB_S * FS)
    timestamps = np.arange(n_cal, dtype=np.float64) / FS
    cfg = fdsi_cbss_config(random_seed=42)
    raw_res = CBSS(cfg).decompose(emg[:n_cal], timestamps)
    try:
        selected = raw_res.select_supervised(gt_binary[:n_cal], roa_th=0.9, fs=FS)
        n_kept = int(selected.spikes.shape[1])
        selected.save(pkl_path)
    except ValueError as err:
        n_kept = 0
        write_new(
            meta_path,
            {"calibrated_at_utc": now(), "record": name, "n_units_kept": 0, "warning": str(err)},
        )
        return name, 0

    write_new(
        meta_path,
        {
            "calibrated_at_utc": now(),
            "record": name,
            "n_units_raw": int(raw_res.spikes.shape[1]),
            "n_units_kept": n_kept,
            "gt_matched_indices": [int(x) for x in selected.gt_matched_indices],
            "calib_roa": [float(x) for x in selected.roa],
            "pkl_sha256": sha256(pkl_path),
        },
    )
    return name, n_kept


def calibrate_all(*, workers: int = 4):
    cases = [
        (subj, cond, snr)
        for subj, _ in CONFIRMATORY_SUBJECTS
        for cond in CONDITIONS
        for snr in SNRS
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_calibrate_one, c) for c in cases]
        for fut in as_completed(futs):
            name, kept = fut.result()
            print(f"[CALIB] {name}: kept={kept} units (RoA>=0.9)", flush=True)


# --------------------------------------------------------------------------------------
# Step 3: Frozen evaluation of Pareto, L2 (3 seeds), P3 (3 seeds), and P3-nomem (3 seeds)
# --------------------------------------------------------------------------------------


def _load_conf_record(subject: str, condition: str, snr: int) -> dict:
    reference()
    from adapt_decomp import CBSSResult

    name = record_stub(subject, condition, snr)
    emg_path = FOLDER / "data" / subject / "noisy" / f"{name}_emg.npz"
    cal_path = FOLDER / "calibration" / subject / f"{name}_cbss.pkl"
    with np.load(emg_path, allow_pickle=False) as arc:
        raw = torch.from_numpy(np.asarray(arc["emg"], dtype=np.float32))
    calibration = CBSSResult.load(cal_path)
    config = fdsi_adapt_config()
    return {
        "raw": raw,
        "calibration": calibration,
        "config": config,
        "subject": subject,
        "condition": condition,
        "snr_db": snr,
        "raw_sha256": sha256(emg_path),
        "calibration_sha256": sha256(cal_path),
    }


def _eval_task(arguments):
    arm_dir, case, spec, disable_memory, threads = arguments
    torch.set_num_threads(threads)
    subject, condition, snr = case
    name = record_stub(subject, condition, snr)
    meta_path = arm_dir / "metadata" / f"{name}.json"
    if meta_path.exists():
        return name

    cal_path = FOLDER / "calibration" / subject / f"{name}_cbss.pkl"
    if not cal_path.exists():
        return f"{name} (skipped: 0 calibrated units)"

    record = _load_conf_record(subject, condition, snr)
    settings = cc.l2_settings()
    if disable_memory:
        settings = make_nomem_settings(settings)

    if spec["kind"] == "pareto":
        adapter = build(record["calibration"], copy.deepcopy(record["config"]))
        pred = replay(record["raw"], adapter, collect_sources=False)
    elif spec["kind"] == "l2":
        ckpt = OUTPUT / spec["checkpoint"]
        adapter = cc.make_l2_adapter(record, ckpt, settings=settings, **spec["parameters"])
        pred = replay(record["raw"], adapter, collect_sources=False)
    else:
        ckpt = OUTPUT / spec["checkpoint"]
        adapter = cc.make_context_adapter(
            record, load_context(ckpt), settings=settings, **spec["parameters"]
        )
        pred = replay(record["raw"], adapter, collect_sources=False)

    pred.pop("sources", None)
    dest = arm_dir / "predictions" / f"{name}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **pred)
    write_new(
        meta_path,
        {
            "predicted_at_utc": now(),
            "record": list(case),
            "spec": spec,
            "disable_memory": disable_memory,
            "files": {dest.relative_to(arm_dir).as_posix(): sha256(dest)},
        },
    )
    return name


def _score_conf_record(arm_dir: Path, case: tuple[str, str, int]) -> dict | None:
    reference()
    from adapt_decomp import CBSSResult
    from adapt_decomp.spikes import rate_of_agreement_paired

    from benchmarks.robust_mvdr.phases import angle_phases, masked

    subject, condition, snr = case
    name = record_stub(subject, condition, snr)
    cal_path = FOLDER / "calibration" / subject / f"{name}_cbss.pkl"
    pred_path = arm_dir / "predictions" / f"{name}.npz"
    if not cal_path.exists() or not pred_path.exists():
        return None

    score_path = arm_dir / "scores" / f"{name}.json"
    if score_path.exists():
        return read(score_path)

    cal = CBSSResult.load(cal_path)
    with np.load(pred_path, allow_pickle=False) as arc:
        binary = np.asarray(arc["spikes"], dtype=np.uint8)
    with np.load(
        FOLDER / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
        allow_pickle=False,
    ) as arc:
        truth = np.asarray(arc["spikes"], dtype=np.uint8)[:, cal.gt_matched_indices]
    with np.load(
        FOLDER / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz",
        allow_pickle=False,
    ) as arc:
        angle = np.asarray(arc["angle"], dtype=np.float32)

    roa, _, lag = rate_of_agreement_paired(truth, binary, fs=FS, tol_spike_ms=2)
    nr, npred = truth.sum(0), binary.sum(0)
    tp = np.rint(roa * (nr + npred) / (1 + roa))
    precision = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    recall = np.divide(tp, nr, out=np.zeros_like(tp), where=nr > 0)

    phases = angle_phases(angle, FS)
    phase_scores = {
        k: (
            rate_of_agreement_paired(masked(truth, m), masked(binary, m), fs=FS, tol_spike_ms=2)[
                0
            ].tolist()
            if m.sum() >= FS
            else [None] * truth.shape[1]
        )
        for k, m in phases.items()
    }
    cycles = ramp_cycles(angle, FS) if condition.startswith("triangular") else []
    cycle_scores = [
        rate_of_agreement_paired(truth[a:b], binary[a:b], fs=FS, tol_spike_ms=2)[0].tolist()
        for a, b in cycles
    ]
    strata = cc.strata_for(truth, angle, binary)
    rows = []
    for j in range(truth.shape[1]):
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "snr_db": snr,
                "unit": j,
                "gt_unit": int(cal.gt_matched_indices[j]),
                "roa": float(roa[j]),
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "lag_samples": int(lag[j]),
                "phases": {k: v[j] for k, v in phase_scores.items()},
                "cycles": [vals[j] for vals in cycle_scores],
                "strata": {k: v[j] for k, v in strata.items()},
            }
        )
    out = {"record": name, "rows": rows}
    write_new(score_path, out)
    return out


def evaluate_all(*, workers: int = 4, threads: int = 2) -> dict:
    """Run all frozen systems on all calibrated confirmatory recordings and score."""
    analysis_path = FOLDER / "analysis.json"
    if analysis_path.exists():
        return read(analysis_path)

    cases = [
        (subj, cond, snr)
        for subj, _ in CONFIRMATORY_SUBJECTS
        for cond in CONDITIONS
        for snr in SNRS
        if (FOLDER / "calibration" / subj / f"{record_stub(subj, cond, snr)}_cbss.pkl").exists()
    ]
    if not cases:
        raise RuntimeError(
            "No calibrated confirmatory records found; run generate and calibrate first"
        )

    l2_h = cc.l2_heads()
    p3_h = cc.final_heads()

    arms: dict[str, tuple[dict, bool]] = {"pareto": ({"kind": "pareto"}, False)}
    for seed in SEEDS:
        arms[f"l2_seed{seed}"] = (l2_h[seed], False)
        arms[f"p3_seed{seed}"] = (p3_h[seed], False)
        arms[f"p3_nomem_seed{seed}"] = (p3_h[seed], True)

    arm_dirs: dict[str, Path] = {}
    for arm_name, (spec, disable_mem) in arms.items():
        arm_dir = FOLDER / "arms" / arm_name
        arm_dirs[arm_name] = arm_dir
        tasks = [(arm_dir, case, spec, disable_mem, threads) for case in cases]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_eval_task, t) for t in tasks]):
                print(f"[EVAL {arm_name}] {fut.result()}", flush=True)

    # Seal every arm before scoring
    for arm_name, arm_dir in arm_dirs.items():
        seal_path = arm_dir / "sealed.json"
        if not seal_path.exists():
            files = {}
            for case in cases:
                meta = read(arm_dir / "metadata" / f"{record_stub(*case)}.json")
                files.update(meta["files"])
            write_new(
                seal_path,
                {"sealed_at_utc": now(), "arm": arm_name, "files": files, "cases": cases},
            )

    # Score all arms
    rows_by_arm: dict[str, dict] = {}
    clusters: dict[tuple[str, str, int], list] = defaultdict(list)
    for arm_name, arm_dir in arm_dirs.items():
        arm_rows = {}
        for case in cases:
            scored = _score_conf_record(arm_dir, case)
            if scored is None:
                continue
            for row in scored["rows"]:
                key = (row["subject"], row["condition"], row["snr_db"], row["unit"])
                arm_rows[key] = row
                if arm_name == "pareto":
                    clusters[(row["subject"], row["condition"], row["gt_unit"])].append(key)
        rows_by_arm[arm_name] = arm_rows

    def _summarize_subset(subset_keys: list) -> dict:
        if not subset_keys:
            return {}
        subset_clusters: dict = defaultdict(list)
        key_set = set(subset_keys)
        for cname, ckeys in clusters.items():
            matched = [k for k in ckeys if k in key_set]
            if matched:
                subset_clusters[cname] = matched

        arm_names = list(rows_by_arm)
        table = {
            "p3_vs_pareto": {**cc.mean_of("p3", 1), "pareto": -1.0},
            "p3_vs_l2": {**cc.mean_of("p3", 1), **cc.mean_of("l2", -1)},
            "p3_nomem_vs_p3": {**cc.mean_of("p3_nomem", 1), **cc.mean_of("p3", -1)},
            "p3_nomem_vs_pareto": {**cc.mean_of("p3_nomem", 1), "pareto": -1.0},
            "l2_vs_pareto": {**cc.mean_of("l2", 1), "pareto": -1.0},
        }
        sub_rows = {a: {k: rows_by_arm[a][k] for k in subset_keys} for a in arm_names}
        point, effects = cc.bootstrap(sub_rows, subset_clusters, table)
        return {
            "n_instances": len(subset_keys),
            "n_clusters": len(subset_clusters),
            "point_values": point,
            "effects": effects,
        }

    all_keys = sorted(rows_by_arm["pareto"])
    matched_keys = [k for k in all_keys if k[1] in MATCHED_CONDITIONS]
    ood_keys = [k for k in all_keys if k[1] in OOD_CONDITIONS]

    result = {
        "analysed_at_utc": now(),
        "scope": (
            "Confirmatory out-of-sample evaluation on 3 brand-new NeuroMotion subjects "
            "(seeds 210601-210603) across matched (ramp10s/ramp5s) and OOD "
            "(ramp3s/sinusoid-0.1Hz) conditions; all models frozen, zero tuning."
        ),
        "calibrated_records": len(cases),
        "all_conditions": _summarize_subset(all_keys),
        "matched_conditions": _summarize_subset(matched_keys),
        "ood_conditions": _summarize_subset(ood_keys),
        "by_condition": {
            cond: _summarize_subset([k for k in all_keys if k[1] == cond]) for cond in CONDITIONS
        },
    }
    write_new(analysis_path, result)
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EMBC 2027 Phase 2: MUniverse NeuroMotion confirmatory & OOD evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate confirmatory subjects via Singularity")
    gen.add_argument("--sif", type=Path, required=True, help="Path to muniverse_neuromotion.sif")

    cal = sub.add_parser("calibrate", help="Run 5-s FDSI CBSS calibration on generated records")
    cal.add_argument("--workers", type=int, default=4)

    ev = sub.add_parser("evaluate", help="Evaluate frozen Pareto/L2/P3/P3-nomem and score")
    ev.add_argument("--workers", type=int, default=4)
    ev.add_argument("--threads", type=int, default=2)

    all_cmd = sub.add_parser("run-all", help="Run generate -> calibrate -> evaluate end to end")
    all_cmd.add_argument("--sif", type=Path, required=True)
    all_cmd.add_argument("--workers", type=int, default=4)
    all_cmd.add_argument("--threads", type=int, default=2)

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_dataset(args.sif)
    elif args.command == "calibrate":
        calibrate_all(workers=args.workers)
    elif args.command == "evaluate":
        res = evaluate_all(workers=args.workers, threads=args.threads)
        print(
            "Phase 2 global P3 vs Pareto:",
            res["all_conditions"]["effects"]["p3_vs_pareto"]["global_roa_pct"],
        )
    elif args.command == "run-all":
        generate_dataset(args.sif)
        calibrate_all(workers=args.workers)
        res = evaluate_all(workers=args.workers, threads=args.threads)
        print(
            "Phase 2 global P3 vs Pareto:",
            res["all_conditions"]["effects"]["p3_vs_pareto"]["global_roa_pct"],
        )


if __name__ == "__main__":
    main()
