"""Reproduce report Table 2 from the authors' FDSI caches and ground-truth spikes.

No calibration, adaptation or hyperparameter search is rerun. Every cached RoA is
independently recomputed with the pinned authors' evaluator before aggregation.
"""

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from download import digest

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a8f53736e297962670276bc123c137050b7056b6"
ARCHIVES = {
    "fdsi-data": ("22882346", "md5:1b146ef935f2bc04a615aa92542e350a"),
    "fdsi-outputs": ("22882323", "md5:75e86229423b8e432391125f7e311908"),
}
# Transcribed targets, deliberately separate from the measured values.
# Columns: mean RoA, median RoA, share >=90%, and mean RoA at 15/20/25/30 dB.
TABLE_2 = {
    "frozen": [37.6, 34.8, 0.0, 39.0, 39.1, 36.9, 37.2],
    "contrast": [72.7, 84.9, 44.3, 77.2, 77.3, 71.7, 70.6],
    "pareto": [79.3, 95.0, 57.7, 81.3, 82.7, 78.2, 78.2],
    "oracle": [84.3, 97.0, 68.5, 84.5, 86.4, 83.3, 84.2],
}
FIGURE_2 = {
    "frozen": [33.6, 38.2, 39.3, 38.5, 38.5],
    "pareto": [91.6, 87.5, 81.0, 72.1, 64.2],
    "oracle": [95.0, 92.2, 91.0, 76.9, 66.8],
}
CONFIGS = {"frozen": None, "contrast": "tpe_sv_median", "pareto": "tpe_pareto", "oracle": "tpe_roa"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty benchmark")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_header(path, key):
    """Read an NPZ member's shape without allocating the full EMG matrix."""
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY header {version}: {path}")
    return shape, dtype


def checked_roa(values, expected_units):
    values = np.asarray(values, dtype=float)
    if values.shape != (expected_units,) or not np.isfinite(values).all():
        raise ValueError("Missing, nonfinite or inconsistent per-unit RoA")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("RoA must be a fraction in [0, 1]")
    return values


def identity_source(cached_indices, calibration_indices):
    """Use the notebook's calibration pairing when an optional cache field is absent.

    No new GT matching is performed. Full-recording RoA must still reproduce the
    cached value for every column before this result can enter the aggregate.
    """
    if cached_indices is None:
        return "calibration; optional cached identity field absent"
    if not np.array_equal(cached_indices, calibration_indices):
        raise ValueError("Cached identities disagree with the published calibration order")
    return "cached identities checked against calibration"


def aggregate(rows, *, expected_recordings=100, expected_units=768):
    """Match notebook pooling over unit-instances, never means of recording means."""
    result = {}
    for method in CONFIGS:
        selected = [row for row in rows if row["method"] == method]
        recordings = {(r["subject"], r["condition"], r["snr_db"]) for r in selected}
        identities = {(r["subject"], r["condition"], r["snr_db"], r["unit"]) for r in selected}
        if (
            len(recordings) != expected_recordings
            or len(selected) != expected_units
            or len(identities) != len(selected)
        ):
            raise ValueError(f"Incomplete or duplicate benchmark rows for {method}")
        values = checked_roa([r["roa"] for r in selected], expected_units) * 100
        snr_means = {}
        for snr in (15, 20, 25, 30):
            subset = [r["roa"] for r in selected if r["snr_db"] == snr]
            if not subset:
                raise ValueError(f"Missing SNR {snr} for {method}")
            snr_means[str(snr)] = float(np.mean(subset) * 100)
        result[method] = {
            "recordings": len(recordings),
            "unit_instances": len(selected),
            "mean_roa_pct": float(np.mean(values)),
            "median_roa_pct": float(np.median(values)),
            "pct_units_roa_ge_90": float(np.mean(values >= 90) * 100),
            "mean_roa_by_snr_pct": snr_means,
            "unit_instances_by_snr": {
                str(snr): sum(r["snr_db"] == snr for r in selected) for snr in (15, 20, 25, 30)
            },
        }
    return result


def compare_table(overall, tolerance_pp=1.0):
    checks = []
    names = ["mean_roa_pct", "median_roa_pct", "pct_units_roa_ge_90"]
    for method, targets in TABLE_2.items():
        observed = [overall[method][key] for key in names]
        observed += [overall[method]["mean_roa_by_snr_pct"][str(s)] for s in (15, 20, 25, 30)]
        for metric, actual, target in zip(
            names + [f"mean_roa_snr{s}_pct" for s in (15, 20, 25, 30)],
            observed,
            targets,
            strict=True,
        ):
            delta = float(actual - target)
            checks.append(
                {
                    "method": method,
                    "metric": metric,
                    "report_pct": target,
                    "observed_pct": actual,
                    "delta_pp": delta,
                    "passed": bool(np.isfinite(delta) and abs(delta) <= tolerance_pp),
                }
            )
    return checks


def verify_downloads(download_dir):
    records = []
    for name, (record_id, expected_checksum) in ARCHIVES.items():
        manifest = json.loads((download_dir / f"{name}_download.json").read_text(encoding="utf-8"))
        if str(manifest["record_id"]) != record_id or len(manifest["files"]) != 1:
            raise ValueError(f"Unexpected Zenodo record for {name}")
        entry = manifest["files"][0]
        archive = download_dir / entry["filename"]
        if archive.parent.resolve() != download_dir.resolve():
            raise ValueError("Invalid archive filename")
        algorithm, expected = expected_checksum.split(":")
        if (
            entry["checksum"] != expected_checksum
            or archive.stat().st_size != entry["size"]
            or digest(archive, algorithm) != expected
        ):
            raise ValueError(f"Official archive checksum mismatch: {archive}")
        records.append({**manifest, "verified_locally": True})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=ROOT / "data/benchmarks/farina")
    parser.add_argument("--reference", type=Path, default=ROOT / "external/farina/adapt_decomp")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/farina/fdsi_cached")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    started = time.perf_counter()
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.reference), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != COMMIT or dirty:
        raise ValueError("Expected the clean pinned v1.0.0 reference checkout")
    sys.path.insert(0, str(args.reference / "src"))
    import torch
    from adapt_decomp import AdaptationResult, CBSSResult
    from adapt_decomp.spikes import rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    torch.set_num_threads(args.threads)
    grid_path = args.reference / "configs/data_configs/fdsi_benchmark_grid.yaml"
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    data_root = args.downloads / "fdsi_benchmark/data"
    cache_root = args.downloads / "fdsi_benchmark/outputs"
    args.output.mkdir(parents=True, exist_ok=True)
    print("Verifying both complete official archives", flush=True)
    downloads = verify_downloads(args.downloads)
    rows, inventory, result_files = [], [], []
    max_difference = 0.0
    for number, (subject, condition, snr) in enumerate(
        product(grid["subjects"], grid["conditions"], grid["snr_levels"]), 1
    ):
        stub = f"{subject}_FDSI_{condition}_snr{snr}dB"
        clean_stub = f"{subject}_FDSI_{condition}"
        noisy_path = data_root / subject / "noisy" / f"{stub}_emg.npz"
        clean_dir = data_root / subject / "clean"
        clean_path = clean_dir / f"{clean_stub}_emg.npz"
        shape, dtype = array_header(noisy_path, "emg")
        clean_shape, _ = array_header(clean_path, "emg")
        n_samples = round((125 if condition == "staircase" else 90) * grid["fs"])
        if shape != (n_samples, 100) or clean_shape != shape or dtype.kind != "f":
            raise ValueError(f"Unexpected EMG shape/dtype: {noisy_path}: {shape}, {dtype}")
        for key in ("angle", "effort"):
            profile_shape, _ = array_header(clean_dir / f"{clean_stub}_{key}.npz", key)
            if profile_shape != (n_samples,):
                raise ValueError(f"Profile sample count differs: {clean_stub}, {key}")
        calibration_path = cache_root / "calibration" / subject / f"{stub}_cbss.pkl"
        calibration = CBSSResult.load(calibration_path)
        indices = np.asarray(calibration.gt_matched_indices)
        n_units = len(indices)
        if (
            n_units == 0
            or indices.shape != (n_units,)
            or len(np.unique(indices)) != n_units
            or calibration.ext_fact != grid["ext_fact"]
            or calibration.spikes.shape != (round(grid["cal_duration_s"] * grid["fs"]), n_units)
            or not np.all(checked_roa(calibration.roa, n_units) >= grid["roa_cal_th"])
        ):
            raise ValueError(f"Inconsistent calibration: {calibration_path}")
        truth = load_gt(clean_dir / f"{clean_stub}_spikes.npz", n_samples=n_samples)
        truth = truth[:, indices]
        inventory.append(
            {
                "subject": subject,
                "condition": condition,
                "snr_db": snr,
                "samples": n_samples,
                "channels": shape[1],
                "calibrated_units": n_units,
                "ground_truth_indices": indices.tolist(),
            }
        )
        result_files.append(
            {
                "path": str(calibration_path.relative_to(args.downloads)),
                "sha256": digest(calibration_path, "sha256"),
            }
        )
        for method, sampler in CONFIGS.items():
            branch = "fixed" if sampler is None else f"lr_fixed/{sampler}"
            suffix = "fixed" if sampler is None else f"lr_fixed_{sampler}"
            path = cache_root / "adaptation" / subject / branch / f"{stub}_adapt_{suffix}.pkl"
            config_path = path.with_name(path.stem + "_config.yaml")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if sampler is None:
                if any(config[key] for key in ("adapt_wh", "adapt_sv", "adapt_sd")):
                    raise ValueError("Frozen comparator unexpectedly adapts")
            elif config["lr_mode"] != "fixed":
                raise ValueError("Table 2 requires the fixed learning-rate mode")
            cached = AdaptationResult.load(path)
            if cached.spikes.shape != (n_samples, n_units):
                raise ValueError(f"Cached unit/sample count mismatch: {path}")
            pairing_source = identity_source(cached.gt_matched_indices, indices)
            saved_roa = checked_roa(cached.roa, n_units)
            recomputed, _, lags = rate_of_agreement_paired(
                truth, cached.spikes.numpy(), fs=grid["fs"], tol_spike_ms=grid["tol_spike_ms"]
            )
            recomputed = checked_roa(recomputed, n_units)
            difference = float(np.max(np.abs(saved_roa - recomputed)))
            max_difference = max(max_difference, difference)
            if difference > 1e-10:
                raise ValueError(f"Cached and recomputed RoA differ by {difference}: {path}")
            for unit, (roa, saved, lag, gt_unit) in enumerate(
                zip(recomputed, saved_roa, lags, indices, strict=True)
            ):
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "snr_db": snr,
                        "method": method,
                        "unit": unit,
                        "gt_unit": int(gt_unit),
                        "roa": float(roa),
                        "cached_roa": float(saved),
                        "lag_samples": int(lag),
                        "identity_source": pairing_source,
                    }
                )
            result_files.append(
                {
                    "path": str(path.relative_to(args.downloads)),
                    "sha256": digest(path, "sha256"),
                    "config_sha256": digest(config_path, "sha256"),
                    "config": config,
                    "identity_source": pairing_source,
                }
            )
            del cached
        print(f"{number}/100: {stub}: {n_units} units, four caches verified", flush=True)

    overall = aggregate(rows)
    checks = compare_table(overall)
    by_condition = []
    for method in CONFIGS:
        for condition in grid["conditions"]:
            subset = [
                r["roa"] for r in rows if r["method"] == method and r["condition"] == condition
            ]
            by_condition.append(
                {
                    "method": method,
                    "condition": condition,
                    "unit_instances": len(subset),
                    "mean_roa_pct": float(np.mean(subset) * 100),
                }
            )
    figure_checks = []
    for method, targets in FIGURE_2.items():
        for condition, target in zip(grid["conditions"], targets, strict=True):
            observed = next(
                r["mean_roa_pct"]
                for r in by_condition
                if r["method"] == method and r["condition"] == condition
            )
            figure_checks.append(
                {
                    "method": method,
                    "condition": condition,
                    "report_pct": target,
                    "observed_pct": observed,
                    "delta_pp": observed - target,
                    "passed": abs(observed - target) <= 1.0,
                }
            )
    passed = all(check["passed"] for check in checks + figure_checks)
    summary = {
        "status": "passed" if passed else "failed",
        "scope": "Report weeks 1-2: download FDSI and reproduce cached baselines",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "reference_commit": revision,
        "reference_clean": not dirty,
        "reference_repository": "https://github.com/imendezguerra/adapt_decomp",
        "reference_notebook": "notebooks/fdsi_benchmark/05_comparison_sv_loss_pareto_roa.ipynb",
        "report_sha256": (
            digest(ROOT / "MUAP_dynamic_DL_research_report.pdf", "sha256")
            if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
            else None
        ),
        "reproduction_script_sha256": digest(Path(__file__), "sha256"),
        "grid_sha256": digest(grid_path, "sha256"),
        "python": sys.version,
        "platform": platform.platform(),
        "threads": args.threads,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "torch", "pyyaml", "pandas", "optuna")
        },
        "downloads": downloads,
        "protocol": {
            "fs_hz": grid["fs"],
            "calibration_seconds": grid["cal_duration_s"],
            "spike_tolerance_ms": grid["tol_spike_ms"],
            "alignment_search_ms": 40,
            "evaluation_window": "full recording, including initial 5 s calibration",
            "alignment": "global per-unit lag recomputed over full recording, authors' evaluator",
            "aggregation": "unweighted mean over calibrated unit-instances, not recording means",
            "learning_rate_mode": "fixed",
            "tuning_subject": grid["optim_sub"],
            "tuning_snr_db": grid["optim_snr"],
            "tuning_conditions": grid["pool_conditions"],
            "oracle": "ground-truth-tuned diagnostic upper comparator; not label-free",
            "execution": "cached predictions evaluated; no model fitting or adaptation rerun",
        },
        "recordings": len(inventory),
        "cached_results_recomputed": 100 * len(CONFIGS),
        "caches_without_identity_field": sum(
            entry.get("identity_source", "").startswith("calibration;") for entry in result_files
        ),
        "cached_config_devices": sorted(
            {entry["config"]["device"] for entry in result_files if "config" in entry}
        ),
        "max_absolute_cached_recomputed_roa_difference": max_difference,
        "overall": overall,
        "by_condition": by_condition,
        "table_2_tolerance_pp": 1.0,
        "table_2_checks": checks,
        "figure_2_checks": figure_checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_csv(args.output / "per_unit.csv", rows)
    write_csv(args.output / "table_2_checks.csv", checks)
    write_csv(args.output / "by_condition.csv", by_condition)
    write_json(args.output / "inventory.json", inventory)
    write_json(args.output / "cache_provenance.json", result_files)
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "overall": overall}, indent=2))
    if not passed:
        raise SystemExit("Report targets differ by more than 1 percentage point; see summary.json")


if __name__ == "__main__":
    main()
