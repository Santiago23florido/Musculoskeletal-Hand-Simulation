"""Execute the original adapt_decomp tutorial, saving actual predictions and scores."""

import argparse
import csv
import dataclasses
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_COMMIT = "8ffeb66ebd30bab9b5167a3fc485c2330a81da0f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmarks/farina/neuromotion")
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "external/farina/adapt_decomp_legacy"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/farina/tutorial_legacy")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compute-loss", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != LEGACY_COMMIT:
        raise ValueError(f"Expected {LEGACY_COMMIT}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(args.reference), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("Reference source has local modifications")
    sys.path.insert(0, str(args.reference / "src"))
    import h5py
    import numpy as np
    import torch
    from adapt_decomp.adaptation import AdaptDecomp
    from adapt_decomp.config import Config
    from adapt_decomp.loaders import load_example
    from adapt_decomp.utils import rate_of_agreement, rate_of_agreement_paired
    from metrics import score_spikes

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading the authors' recording and MATLAB calibration", flush=True)
    path_emg = args.data / "data/data_sim.hdf5"
    path_calib = args.data / "data/calibration/decomp_sim.mat"
    data = load_example(path_emg, path_calib, False)
    fs = data["fs"]
    truth = data["spikes_gt"].numpy()
    calib_spikes = data["spikes_calib"].numpy()
    n_cal = calib_spikes.shape[0]
    # These labels are used exclusively for evaluation, never by AdaptDecomp.
    _, pairs, lags = rate_of_agreement(truth[:n_cal], calib_spikes, fs=fs, tol_spike_ms=2)
    if [pair[1] for pair in pairs] != list(range(calib_spikes.shape[1])):
        raise ValueError("Calibration did not match every output unit")
    gt_indices = np.array([pair[0] for pair in pairs])
    paired_truth = truth[:, gt_indices]
    angle = data["angle_profile"].numpy().ravel()
    n_samples = len(angle)
    windows = [("full", 0, n_samples), ("held_out", n_cal, n_samples)]
    # The released tutorial is a monotonic staircase, with 30-second plateaus.
    for value in np.unique(np.round(angle)).tolist():
        if value % 10 != 0:
            continue
        indices = np.flatnonzero(np.isclose(angle, value, atol=1e-5))
        if len(indices) >= fs * 10:
            windows.append((f"plateau_{value:g}deg", int(indices[0]), int(indices[-1] + 1)))
    summary = {
        "status": "tutorial_execution_only; full paper reproduction requires all original data",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "reference_commit": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "seed": 0,
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "h5py", "torch")
        },
        "data_doi": "10.5281/zenodo.22880910",
        "fs_hz": fs,
        "samples": n_samples,
        "channels": int(data["emg"].shape[1]),
        "output_units": int(calib_spikes.shape[1]),
        "calibration_samples": n_cal,
        "calibration_provenance": "provided MATLAB result, not recomputed",
        "preprocess": False,
        "dc_centering": "legacy Data uses the full-recording mean even with preprocess=False",
        "spike_tolerance_ms": 2,
        "alignment": "independent scores freeze unit identities and lags from MATLAB calibration",
        "expected_notebook_whole_recording_roa_pct": {
            "fixed": {"mean": 44.31, "std": 22.74, "median": 39.37},
            "adapted": {"mean": 90.25, "std": 19.93, "median": 98.58},
        },
        "methods": {},
    }
    with h5py.File(path_emg) as handle:
        summary["hdf5_attributes"] = {str(k): str(v) for k, v in handle.attrs.items()}
    rows = []
    for method, adapt in (("fixed", False), ("adapted", True)):
        config = Config(device="cpu")
        config.ext_fact = data["ext_fact"]
        config.adapt_wh = config.adapt_sv = config.adapt_sd = adapt
        config.compute_loss = args.compute_loss
        config.save_params = False
        print(f"Running {method}, {n_samples / fs:.1f} seconds of EMG", flush=True)
        started = time.perf_counter()
        # Clone every mutable input as in the authors' notebook.
        tensor_names = (
            "emg",
            "whitening",
            "sep_vectors",
            "base_centr",
            "spikes_centr",
            "emg_calib",
            "ipts_calib",
            "spikes_calib",
        )
        model = AdaptDecomp(
            **{name: data[name].clone() for name in tensor_names},
            preprocess=False,
            config=config,
        )
        output = model.run()
        elapsed = time.perf_counter() - started
        arrays = {key: value.detach().cpu().numpy() for key, value in output.items()}
        if not np.isfinite(arrays["ipts"]).all():
            raise ValueError(f"Nonfinite source estimates in {method}")
        np.savez_compressed(args.output / f"{method}.npz", **arrays)
        # Tutorial metric: same identity across methods, realignment on the full trace.
        roa, _, _ = rate_of_agreement_paired(paired_truth, arrays["spikes"], fs=fs, tol_spike_ms=2)
        actual = {
            "mean": float(roa.mean() * 100),
            "std": float(roa.std() * 100),
            "median": float(np.median(roa) * 100),
        }
        target = summary["expected_notebook_whole_recording_roa_pct"][method]
        timing = arrays["total_time_ms"]
        summary["methods"][method] = {
            "config": dataclasses.asdict(config),
            "wall_seconds_including_initialization": elapsed,
            "upstream_batch_time_ms_median": float(np.median(timing)),
            "upstream_batch_time_ms_p95": float(np.percentile(timing, 95)),
            "notebook_roa_pct": actual,
            "difference_from_notebook_percentage_points": {
                k: actual[k] - target[k] for k in target
            },
            "notebook_match_within_0_1_percentage_point": all(
                abs(actual[k] - target[k]) <= 0.1 for k in target
            ),
        }
        for label, start, stop in windows:
            for unit, (_, prediction_index) in enumerate(pairs):
                scores = score_spikes(
                    np.flatnonzero(paired_truth[:, unit]),
                    np.flatnonzero(arrays["spikes"][:, prediction_index]),
                    tolerance=round(fs * 0.002),
                    lag=int(lags[unit]),
                    start=start,
                    stop=stop,
                )
                rows.append(
                    {
                        "method": method,
                        "window": label,
                        "unit": unit,
                        "gt_unit": int(gt_indices[unit]),
                        "lag_samples": int(lags[unit]),
                        "start_s": start / fs,
                        "stop_s": stop / fs,
                        **scores,
                    }
                )
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            f"{method}: notebook RoA mean={actual['mean']:.3f}%, "
            f"median={actual['median']:.3f}%, wall={elapsed:.1f}s",
            flush=True,
        )
        del model, output, arrays
        gc.collect()
    with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "evaluation_reference.npz",
        spikes=paired_truth,
        ground_truth_indices=gt_indices,
        alignment_samples=np.asarray(lags),
        angle_deg=angle,
        fs=fs,
    )
    print(f"Saved actual predictions, metrics and provenance to {args.output}", flush=True)


if __name__ == "__main__":
    main()
