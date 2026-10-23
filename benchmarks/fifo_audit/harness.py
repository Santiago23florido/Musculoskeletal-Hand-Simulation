"""Replay the real detector in the 512-dimensional, ten-batch FIFO regime.

The frozen decoders are controls, not edited dependencies. All ablations are here,
outside robust_mvdr's registered source set. No parameter is fitted to outcomes.
"""

import argparse
import copy
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.pareto_residual.detector import Detector
from benchmarks.robust_mvdr.common import FIFO_BATCHES, SUBTRACTION_FLOOR, source_hashes
from benchmarks.robust_mvdr.filters import lag_matrix, robust_filters
from benchmarks.robust_mvdr.synthetic.generator import (
    ElectrodeGrid,
    MuapModel,
    discharge_trains,
    sample_motor_unit_pool,
)
from benchmarks.robust_mvdr.synthetic.study import all_lags
from benchmarks.stage_a.common import ROOT, sha256
from benchmarks.stage_b.data import windows_mean
from benchmarks.stage_b.inference import analytic_filters, extended_template

FS = 2048
CAL_SECONDS = 5


@dataclass(frozen=True)
class Variant:
    alpha: float = 0.0
    normalization: str = "unit_response"
    in_sample: bool = True
    rho: float = 0.0
    loading: float = 0.001


VARIANTS = {
    "stage_b_static": Variant(alpha=0.5, normalization="output_power"),
    "static_fifo_mvdr": Variant(),
    "static_subtract": Variant(rho=0.5, loading=0.03),
    "fifo_mvdr_output_power": Variant(normalization="output_power"),
    "fifo_mvdr_out_of_sample": Variant(in_sample=False),
    "stage_b_out_of_sample": Variant(alpha=0.5, normalization="output_power", in_sample=False),
    "subtract_out_of_sample": Variant(rho=0.5, loading=0.03, in_sample=False),
}


def length_at(seconds, minimum=1.0):
    """8--13 s shortening, hold until 18 s, 18--23 s return."""
    down = np.clip((np.asarray(seconds) - 8.0) / 5.0, 0.0, 1.0)
    up = np.clip((np.asarray(seconds) - 18.0) / 5.0, 0.0, 1.0)
    return 1.0 - (1.0 - minimum) * (down - up)


@torch.no_grad()
def synthetic_record(*, seed=3, units=8, dimension=512, seconds=25, snr_db=20, minimum=1.0):
    """Calibration-only PCA/STA, true discharges, and independently rendered EMG.

    Separability is peak calibration-whitened MUAP energy. All active units are
    rendered, including those not selected for calibration. Length-dependent MUAPs
    use 33 fixed length knots, interpolated at each discharge (no label fitting).
    """
    from adapt_decomp import CBSSResult
    from adapt_decomp.adaptation import AdaptConfig
    from adapt_decomp.preprocessing import extend_data

    if seconds <= CAL_SECONDS or not 1 <= dimension <= 1000 or not 0 < minimum <= 1:
        raise ValueError("Need >5 seconds, 1--1000 PCA axes, and 0 < minimum <= 1")
    rng = np.random.default_rng(seed)
    grid = ElectrodeGrid()
    pool = sample_motor_unit_pool(64, rng)
    model = MuapModel(grid)
    n, n_cal = round(seconds * FS), CAL_SECONDS * FS
    trains = discharge_trains(pool, 0.15, n, FS, rng)
    active = [u for u, train in enumerate(trains) if len(train)]
    cal_muaps = {u: model.unit_muap(pool, u, 1.0).reshape(100, 64) for u in active}
    clean = np.zeros((n + 64, 100), dtype=np.float64)
    for u in active:
        if minimum == 1.0:
            bank = cal_muaps[u][None]
        else:
            bank = np.stack(
                [model.unit_muap(pool, u, s).reshape(100, 64) for s in np.linspace(1, minimum, 33)]
            )
        for discharge in trains[u]:
            position = (
                float((1 - length_at(discharge / FS, minimum)) / (1 - minimum) * 32)
                if minimum < 1
                else 0.0
            )
            lower = min(int(position), len(bank) - 1)
            upper = min(lower + 1, len(bank) - 1)
            waveform = bank[lower] * (1 - (position - lower)) + bank[upper] * (position - lower)
            clean[discharge : discharge + 64] += waveform.T
    clean = clean[:n]
    noise_sd = np.sqrt(np.mean(clean[:n_cal] ** 2, axis=0) / 10 ** (snr_db / 10))
    emg = torch.from_numpy((clean + rng.normal(size=clean.shape) * noise_sd).astype(np.float32))
    extended_cal = extend_data(emg[:n_cal], 10)
    extended_cal -= extended_cal.mean(0)
    covariance = extended_cal.T @ extended_cal / n_cal
    _, axes = torch.linalg.eigh(covariance)
    axes = axes[:, -dimension:]
    projected = extended_cal @ axes
    projection = axes / projected.square().mean(0).sqrt().clamp_min(1e-6)
    x_cal = extended_cal @ projection
    ranked = []
    for u in active:
        lag_vectors = torch.from_numpy(all_lags(cal_muaps[u])).float() @ projection
        energy = lag_vectors.square().sum(1)
        peak = int(energy[10:64].argmax()) + 10
        ranked.append((float(energy[peak]), u, peak))
    selected = sorted(ranked, reverse=True)[:units]
    truth = np.zeros((n, len(selected)), dtype=np.uint8)
    for j, (_, u, peak) in enumerate(selected):
        times = trains[u] + peak
        truth[times[times < n], j] = 1
    templates = torch.stack(
        [windows_mean(emg[:n_cal], np.flatnonzero(truth[:n_cal, j])) for j in range(len(selected))]
    )
    amplitude = templates.flatten(1).norm(dim=1).clamp_min(1e-7)
    template = (templates / amplitude[:, None, None]).reshape(-1, 10, 10, 64)
    template = template.permute(0, 3, 1, 2)[:, None].contiguous()
    config = AdaptConfig(fs=FS, device="cpu", compute_loss=False, debug=False)
    signal = extend_data(emg, 10)
    for start in range(0, len(signal), config.batch_size):
        trim = config.ext_fact if start == 0 else 0
        block = signal[start + trim : start + config.batch_size]
        block -= block.mean(0)
    signal[: config.ext_fact] = 0
    x = signal @ projection
    calibration = CBSSResult(
        sources=np.zeros((n_cal, len(selected)), dtype=np.float32),
        spikes=truth[:n_cal].copy(),
        spikes_dict={},
        sep_vectors=np.eye(dimension, len(selected), dtype=np.float32),
        whitening=np.eye(dimension, dtype=np.float32),
        extension_mean=np.zeros((1, dimension), dtype=np.float32),
        spikes_centr=np.ones(len(selected), dtype=np.float32),
        base_centr=np.full(len(selected), 0.01, dtype=np.float32),
        sil=np.full(len(selected), 0.9, dtype=np.float32),
        cov_isi=np.full(len(selected), 0.1, dtype=np.float32),
        ext_fact=10,
        emg=emg[:n_cal].numpy(),
    )
    batches = (n + config.batch_size - 1) // config.batch_size
    metadata = dict(
        seed=seed,
        pool_units=64,
        active_units=len(active),
        excitation=0.15,
        selected_units=[u for _, u, _ in selected],
        peak_delays=[p for _, _, p in selected],
        dimension=dimension,
        seconds=seconds,
        snr_db=snr_db,
        minimum_length=minimum,
        length_knots=33,
        selection="greatest calibration-PCA-whitened MUAP energy; peak searched at lags 10--63",
        centering="global calibration mean for x_cal; per-batch mean for x; first 10 samples zero",
        templates="calibration EMG STA, radius 32; no post-calibration truth in decoder",
        preprocessing="generator EMG, no additional temporal filtering (as the synthetic fixture)",
    )
    return dict(
        x=x,
        x_cal=x_cal,
        features=torch.zeros(batches, 120),
        q=torch.zeros(batches),
        template=template,
        amplitude=amplitude,
        projection=projection,
        calibration=calibration,
        config=config,
        metadata=metadata,
    ), truth


def _filters(covariance, record, lags, lag0, rate, variant):
    if variant.normalization == "output_power":
        if variant.rho:
            raise ValueError("Output-power subtraction is not one of the specified ablations")
        h = extended_template(record["template"], record["projection"], record["amplitude"])
        return analytic_filters(covariance, h, rate, alpha=variant.alpha, ridge=variant.loading)
    if variant.normalization != "unit_response":
        raise ValueError("Unknown normalization")
    return robust_filters(
        covariance,
        lags,
        lag0,
        rate,
        rho=variant.rho,
        loading=variant.loading,
        alpha=variant.alpha,
        floor=SUBTRACTION_FLOOR,
    )


def _initial_decoder(record, variant, lags, lag0):
    calibration = record["calibration"]
    rate = torch.from_numpy(calibration.spikes).float().mean(0)
    covariance = record["x_cal"].T @ record["x_cal"] / len(record["x_cal"])
    filters = _filters(covariance, record, lags, lag0, rate, variant)
    sources = record["x_cal"] @ filters
    spikes = torch.from_numpy(calibration.spikes).bool()
    peak = torch.stack(
        [sources[spikes[:, j], j].square().mean().sqrt() for j in range(spikes.shape[1])]
    ).clamp_min(1e-6)
    gain = torch.from_numpy(calibration.spikes_centr).float().sqrt() / peak
    sources = sources * gain
    calibrated = copy.copy(calibration)
    calibrated.sources = sources.numpy()
    calibrated.spikes_centr = np.asarray(
        [float(sources[spikes[:, j], j].square().mean()) for j in range(spikes.shape[1])],
        dtype=np.float32,
    )
    calibrated.base_centr = np.asarray(
        [float(sources[~spikes[:, j], j].square().mean()) for j in range(spikes.shape[1])],
        dtype=np.float32,
    )
    return calibrated, filters, gain, covariance, rate


@torch.no_grad()
def decode(record, variant):
    """The static loop, differing only in alpha, normalization, and FIFO ordering."""
    if str(record["config"].device) != "cpu" or record["x"].device.type != "cpu":
        raise ValueError("The audit only accepts CPU records")
    lags, lag0 = lag_matrix(record["template"], record["projection"], record["amplitude"])
    calibrated, filters0, gain, covariance0, rate = _initial_decoder(record, variant, lags, lag0)
    config = record["config"]
    detector = Detector(calibrated, config)
    threshold_calibration = (
        ((detector.decomp.spikes_centr + detector.decomp.base_centr) * 0.5).sqrt().tolist()
    )
    fifo = deque(maxlen=FIFO_BATCHES)
    for start in range(max(0, len(record["x_cal"]) - 2040), len(record["x_cal"]), 204):
        block = record["x_cal"][start : start + 204]
        fifo.append(block.T @ block / len(block))
    predictions = np.zeros((len(record["x"]), record["template"].shape[0]), dtype=np.uint8)
    sources_out = np.zeros_like(predictions, dtype=np.float32)
    durations = []
    for index, start in enumerate(range(0, len(record["x"]), config.batch_size)):
        tick = time.perf_counter()
        stop = min(start + config.batch_size, len(record["x"]))
        trim = config.ext_fact if index == 0 else 0
        block = record["x"][start + trim : stop]
        if start < CAL_SECONDS * config.fs:
            filters = filters0
        else:
            current = block.T @ block / len(block)
            if variant.in_sample:
                fifo.append(current)
            covariance = torch.stack(list(fifo)).mean(0) if fifo else covariance0
            filters = _filters(covariance, record, lags, lag0, rate, variant)
            if not variant.in_sample:
                fifo.append(current)
        sources = block @ filters * gain
        binary, trusted = detector.step(sources, index)
        rate = 0.95 * rate + 0.05 * trusted.float().mean(0)
        predictions[start + trim : stop] = binary.numpy()
        sources_out[start + trim : stop] = sources.numpy()
        durations.append((time.perf_counter() - tick) * 1000)
    predictions[: CAL_SECONDS * config.fs] = record["calibration"].spikes
    return dict(
        spikes=predictions,
        sources=sources_out,
        timing={"median_core_batch_ms": float(np.median(durations))},
        detector=dict(
            absolute_source_threshold_calibration=threshold_calibration,
            absolute_source_threshold_final=(
                (detector.decomp.spikes_centr + detector.decomp.base_centr) * 0.5
            )
            .sqrt()
            .tolist(),
            first_fifo_batch_start_seconds=(
                (CAL_SECONDS * config.fs + config.batch_size - 1)
                // config.batch_size
                * config.batch_size
                / config.fs
            ),
        ),
    )


def amplitude_metrics(sources, truth, fs, windows):
    """One lag per unit, learned only from mean absolute source in 1--5 s."""
    lags = []
    per_unit = {key: [] for key in windows}
    for unit in range(truth.shape[1]):
        times = np.flatnonzero(truth[:, unit])
        calibration = times[(times >= fs) & (times < CAL_SECONDS * fs)]
        if not len(calibration):
            raise ValueError("No calibration truth for amplitude alignment")
        offsets = range(-15, 16)
        lag = max(offsets, key=lambda d: np.abs(sources[calibration + d, unit]).mean())
        lags.append(lag)
        for name, (start, stop) in windows.items():
            selected = times[(times >= round(start * fs)) & (times < round(stop * fs))] + lag
            selected = selected[(selected >= 0) & (selected < len(sources))]
            per_unit[name].append(
                float(np.median(np.abs(sources[selected, unit]))) if len(selected) else None
            )
    baseline = np.asarray(per_unit["calibration"], dtype=float)
    result = {"lag_samples": lags, "per_unit_median_abs_source": per_unit}
    for name, values in per_unit.items():
        array = np.asarray(values, dtype=float)
        valid = np.isfinite(array) & np.isfinite(baseline) & (baseline > 0)
        result[name] = dict(
            median=float(np.median(array[valid])) if valid.any() else None,
            median_ratio_vs_calibration=float(np.median(array[valid] / baseline[valid]))
            if valid.any()
            else None,
            per_unit_ratio_vs_calibration=[
                float(a / b) if np.isfinite(a) and b > 0 else None
                for a, b in zip(array, baseline, strict=True)
            ],
        )
    return result


def score_window(truth, predicted, fs, start_seconds):
    from adapt_decomp.spikes import rate_of_agreement_paired

    start = round(start_seconds * fs)
    ref, pred = truth[start:], predicted[start:]
    roa, _, lag = rate_of_agreement_paired(ref, pred, fs=fs, tol_spike_ms=2)
    nr, npred = ref.sum(0).astype(float), pred.sum(0).astype(float)
    tp = np.round(roa * (nr + npred) / (1 + roa))
    return dict(
        start_seconds=start_seconds,
        roa_mean=float(np.mean(roa)),
        roa_per_unit=roa.tolist(),
        recall_mean=float(np.mean(np.divide(tp, nr, out=np.zeros_like(tp), where=nr > 0))),
        precision_mean=float(np.mean(np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0))),
        true_positives=int(tp.sum()),
        false_positives=int((npred - tp).sum()),
        false_negatives=int((nr - tp).sum()),
        alignment_lag_samples=lag.tolist(),
    )


def compare_predictions(actual, expected):
    """Abort on a replay mismatch; a refactor must reproduce the frozen decoder."""
    np.testing.assert_allclose(actual["sources"], expected["sources"], rtol=1e-4, atol=1e-6)
    agreement = float(np.mean(actual["spikes"] == expected["spikes"]))
    if agreement < 0.999:
        raise AssertionError(f"Spike sample agreement {agreement:.9f} < 0.999")
    return dict(
        source_rtol=1e-4,
        source_atol=1e-6,
        spike_sample_agreement=agreement,
        max_source_abs_difference=float(np.max(np.abs(actual["sources"] - expected["sources"]))),
    )


def provenance():
    return dict(
        registered_source_sha256=source_hashes(),
        audit_source_sha256={
            str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        external_reference_commit="a8f53736e297962670276bc123c137050b7056b6",
        cpu_only=True,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        threads=torch.get_num_threads(),
    )


def write_new_json(path, value):
    """Audit results are new files, never overwrites of frozen experiment artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _synthetic_scenario(name, snr, minimum, score_start, seed=3):
    print("prepare", name, "seed", seed, flush=True)
    record, truth = synthetic_record(seed=seed, snr_db=snr, minimum=minimum)
    result = dict(metadata=record["metadata"], score_start_seconds=score_start, methods={})
    for method, variant in VARIANTS.items():
        predicted = decode(record, variant)
        metrics = score_window(truth, predicted["spikes"], FS, score_start)
        amplitude = amplitude_metrics(
            predicted["sources"], truth, FS, dict(calibration=(1, 5), after=(score_start, 25))
        )
        result["methods"][method] = dict(
            score=metrics,
            amplitude=amplitude,
            timing=predicted["timing"],
            detector=predicted["detector"],
        )
        if method in ("static_fifo_mvdr", "stage_b_static"):
            result["methods"][method]["per_unit_amplitude_vs_roa"] = [
                dict(
                    unit=j,
                    pool_unit=record["metadata"]["selected_units"][j],
                    amplitude_ratio=amplitude["after"]["per_unit_ratio_vs_calibration"][j],
                    calibration_amplitude=amplitude["per_unit_median_abs_source"]["calibration"][j],
                    after_amplitude=amplitude["per_unit_median_abs_source"]["after"][j],
                    roa=metrics["roa_per_unit"][j],
                )
                for j in range(truth.shape[1])
            ]
        print(
            name,
            method,
            "RoA",
            round(metrics["roa_mean"], 6),
            "amplitude ratio",
            round(amplitude["after"]["median_ratio_vs_calibration"], 6),
            flush=True,
        )
    return result


def qualitative_claims(cases):
    """Evaluate amplitude claims only; no reference RoA or fitted cutoff curve."""
    c1, c2, units = [], [], []
    pairs = (
        ("fifo_mvdr_out_of_sample", "static_fifo_mvdr"),
        ("stage_b_out_of_sample", "stage_b_static"),
        ("subtract_out_of_sample", "static_subtract"),
    )
    for name, result in cases.items():
        methods = result["methods"]

        def ratio(method, methods=methods):
            return methods[method]["amplitude"]["after"]["median_ratio_vs_calibration"]

        c1.append(
            dict(
                case=name,
                fifo_ratio=ratio("static_fifo_mvdr"),
                stage_b_ratio=ratio("stage_b_static"),
                holds=ratio("static_fifo_mvdr") < ratio("stage_b_static"),
            )
        )
        for out, inside in pairs:
            c2.append(
                dict(
                    case=name,
                    variant=out,
                    in_sample_ratio=ratio(inside),
                    out_of_sample_ratio=ratio(out),
                    holds=ratio(out) > ratio(inside),
                )
            )
        for method in ("static_fifo_mvdr", "stage_b_static"):
            units.extend(
                dict(case=name, method=method, **row)
                for row in methods[method]["per_unit_amplitude_vs_roa"]
            )
    return dict(
        C1=dict(holds_all=all(row["holds"] for row in c1), comparisons=c1),
        C2=dict(holds_all=all(row["holds"] for row in c2), comparisons=c2),
        C3=dict(
            evaluation="per-unit description; no fitted curve or assumed universal cutoff",
            per_unit=units,
        ),
        unique_cases=len(cases),
    )


def run_synthetic(output):
    tick = time.perf_counter()
    report = dict(
        started_at_utc=datetime.now(UTC).isoformat(),
        provenance=provenance(),
        variants={name: asdict(v) for name, v in VARIANTS.items()},
        scenarios={},
        sensitivity={},
        declared_claims=dict(
            C1="FIFO unit-response median amplitude ratio < Stage B ratio in every case and seed",
            C2="Every out-of-sample median amplitude ratio > its in-sample counterpart",
            C3="Describe per-unit amplitude versus RoA for a cutoff; fit no curve",
            roa="RoA magnitudes and rankings are observations, not confirmation targets",
        ),
        interpretation="diagnostic synthetic attribution, not a new outer-subject experiment",
        scoring="adapt_decomp rate_of_agreement_paired, 2 ms; means over the same eight units",
    )
    scenarios = (
        ("stationary_20db", 20, 1.0, 5),
        ("stationary_30db", 30, 1.0, 5),
        ("ramp_094", 20, 0.94, 8),
        ("ramp_091", 20, 0.91, 8),
    )
    for name, snr, minimum, score_start in scenarios:
        report["scenarios"][name] = _synthetic_scenario(name, snr, minimum, score_start)
    unique = dict(report["scenarios"])
    for snr in (20, 30):
        for seed in (3, 4, 5):
            name = f"stationary_{snr}db_seed{seed}"
            if seed == 3:
                # Reuse the identical completed record, rather than score it twice.
                result = report["scenarios"][f"stationary_{snr}db"]
            else:
                result = _synthetic_scenario(name, snr, 1.0, 5, seed=seed)
                unique[name] = result
            report["sensitivity"][name] = result
    report["qualitative_claims"] = qualitative_claims(unique)
    report["elapsed_seconds"] = time.perf_counter() - tick
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    destination = Path(output) / "fifo_audit_synthetic.json"
    write_new_json(destination, report)
    print("saved", destination, "seconds", report["elapsed_seconds"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_synthetic(args.output)


if __name__ == "__main__":
    main()
