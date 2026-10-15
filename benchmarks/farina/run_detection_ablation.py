"""Reproduce the cached notebook comparator using fixed sources and adaptive centroids."""

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
LEGACY_COMMIT = "8ffeb66ebd30bab9b5167a3fc485c2330a81da0f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/farina/tutorial_legacy")
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmarks/farina/neuromotion")
    args = parser.parse_args()
    reference = ROOT / "external/farina/adapt_decomp_legacy"
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != LEGACY_COMMIT or dirty:
        raise ValueError("The reference must be the unmodified pinned legacy checkout")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    if summary["reference_commit"] != revision:
        raise ValueError("Cached sources came from a different implementation")
    if any(
        summary["methods"]["fixed"]["config"][flag] for flag in ("adapt_wh", "adapt_sv", "adapt_sd")
    ):
        raise ValueError("Cached sources must come from the fully fixed condition")
    sys.path.insert(0, str(reference / "src"))
    import numpy as np
    import torch
    from adapt_decomp.adaptation import AdaptDecomp
    from adapt_decomp.config import Config
    from adapt_decomp.utils import rate_of_agreement_paired
    from scipy.io import loadmat

    with np.load(args.run / "fixed.npz", allow_pickle=False) as archive:
        sources = archive["ipts"]
    with np.load(args.run / "evaluation_reference.npz", allow_pickle=False) as archive:
        truth = archive["spikes"]
    calibration = loadmat(args.data / "data/calibration/decomp_sim.mat", simplify_cells=True)
    # spike_det depends only on these centroids, config and unit count. Reuse the
    # fixed projections to isolate detection without rerunning identical filters.
    detector = AdaptDecomp.__new__(AdaptDecomp)
    detector.config = Config(
        **{
            key: value
            for key, value in summary["methods"]["fixed"]["config"].items()
            if key != "spike_dist"
        }
    )
    detector.config.adapt_sd = True
    detector.units = sources.shape[1]
    base = calibration["N_CENT"].copy()
    spike = calibration["SIG_CENT"].copy()
    detector.decomp = SimpleNamespace(
        base_centr=base, spikes_centr=spike, height=(base + spike) / 2
    )
    predictions = np.zeros(sources.shape, dtype=np.int32)
    for start in range(0, len(sources), detector.config.batch_size):
        stop = min(len(sources), start + detector.config.batch_size)
        predictions[start:stop] = detector.spike_det(torch.from_numpy(sources[start:stop])).numpy()
    roa, _, _ = rate_of_agreement_paired(truth, predictions, fs=summary["fs_hz"], tol_spike_ms=2)
    actual = {
        "mean": float(roa.mean() * 100),
        "std": float(roa.std() * 100),
        "median": float(np.median(roa) * 100),
    }
    expected = summary["expected_notebook_whole_recording_roa_pct"]["fixed"]
    result = {
        "purpose": "isolate detection adaptation with whitening and separation fixed",
        "reference_commit": revision,
        "config": dataclasses.asdict(detector.config),
        "notebook_roa_pct": actual,
        "difference_from_cached_notebook_comparator_percentage_points": {
            key: actual[key] - expected[key] for key in expected
        },
        "matches_cached_notebook_within_0_1_percentage_point": all(
            abs(actual[key] - expected[key]) <= 0.1 for key in expected
        ),
        "runtime_note": "cached source reuse; no end-to-end timing claim",
        "interpretation": "cached notebook comparator is reproduced when only adapt_sd is enabled",
    }
    np.savez_compressed(args.run / "fixed_filters_adaptive_detection.npz", spikes=predictions)
    (args.run / "detection_ablation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
