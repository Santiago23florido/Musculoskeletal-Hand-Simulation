"""Register, predict every outer record, seal, then score against the same 591 instances.

The protocol is registered before any prediction. All predictions are hashed and
sealed before the first true spike train is opened. Stage B's frozen morph and
observer weights use the original model classes and verified frozen provenance;
nothing is trained here and historical registrations remain unchanged.
"""

import json
import time
from datetime import UTC, datetime

import numpy as np
import torch

from benchmarks.stage_b.common import OUTPUT as STAGE_B_OUTPUT
from benchmarks.stage_b.data import prepare

from . import frozen
from .common import (
    ANGLE_REST_DEG,
    CONDITIONS,
    DATA,
    FIFO_BATCHES,
    LAG_KNOTS,
    METHODS,
    OUTPUT,
    PRIMARY,
    REFERENCE_METHODS,
    SPEED_REST_DEG_S,
    SUBJECTS,
    SUBTRACTION_FLOOR,
    pareto_path,
    sha256,
    source_hashes,
    stub,
    write_json,
)
from .inference import lag_tables, predict
from .phases import angle_phases, masked

SNRS = (30, 25, 20, 15)
RECORDS = [(s, c, n) for s in SUBJECTS[1:] for c in CONDITIONS for n in SNRS]


def specification():
    return dict(
        methods=METHODS,
        primary=PRIMARY,
        lag_knots=LAG_KNOTS,
        fifo_batches=FIFO_BATCHES,
        subtraction_floor=SUBTRACTION_FLOOR,
        angle_rest_deg=ANGLE_REST_DEG,
        speed_rest_deg_s=SPEED_REST_DEG_S,
        reference_methods=list(REFERENCE_METHODS),
        detector="original SD detector via benchmarks.pareto_residual.detector",
        trained_parameters=0,
        labels_used_before_scoring=False,
    )


def register():
    path = OUTPUT / "protocol.json"
    if path.exists():
        raise ValueError("Protocol already exists; use a new ROBUST_MVDR_OUTPUT")
    locked = STAGE_B_OUTPUT / "locked.json"
    write_json(
        path,
        dict(
            registered_at_utc=datetime.now(UTC).isoformat(),
            spec=specification(),
            source_sha256=source_hashes(),
            stage_b_locked_sha256=sha256(locked) if locked.exists() else None,
            stage_b_frozen_provenance=frozen.validate(STAGE_B_OUTPUT) if _needs_models() else None,
        ),
    )


def checked():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if protocol["spec"] != json.loads(json.dumps(specification())):
        raise ValueError("Specification changed after registration")
    if protocol["source_sha256"] != source_hashes():
        raise ValueError("Code changed after registration; start a new output directory")
    if _needs_models() and protocol["stage_b_frozen_provenance"] != frozen.validate(STAGE_B_OUTPUT):
        raise ValueError("Stage B frozen provenance changed after registration")
    return protocol


def _needs_models():
    return any(spec["mode"] != "static" for spec in METHODS.values())


@torch.no_grad()
def predict_subject(subject):
    protocol = checked()
    if subject not in SUBJECTS[1:] or (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Invalid outer subject or scoring has started")
    models = []
    if _needs_models():
        models = frozen.load_models(
            subject, STAGE_B_OUTPUT, expected=protocol["stage_b_frozen_provenance"]
        )
        if protocol["stage_b_locked_sha256"] != sha256(STAGE_B_OUTPUT / "locked.json"):
            raise ValueError("Stage B frozen models differ from the registered ones")
    from adapt_decomp.spikes import get_sil

    for record_subject, condition, snr in RECORDS:
        if record_subject != subject:
            continue
        name = stub(subject, condition, snr)
        done = OUTPUT / "predictions/metadata" / f"{name}.json"
        if done.exists():
            continue
        tick = time.perf_counter()
        record = prepare(subject, condition, snr)
        tables = lag_tables(record, models) if models else None
        metadata = dict(methods={}, prediction_sha256={})
        for method, spec in METHODS.items():
            result = predict(record, models, spec, tables=tables)
            sil = get_sil(
                torch.from_numpy(result["sources"]),
                torch.from_numpy(result["spikes"].astype(np.int32)),
                record["config"].spike_min_dist,
                peak_power=2,
            ).numpy()
            path = OUTPUT / "predictions" / method / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, spikes=result["spikes"], q=result["q"], sil=sil)
            metadata["prediction_sha256"][method] = sha256(path)
            metadata["methods"][method] = result["timing"]
        metadata.update(
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            completed_at_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - tick,
            outer_true_spikes_loaded=False,
        )
        write_json(done, metadata)
        print("predicted", name, round(metadata["elapsed_seconds"], 1), flush=True)


def seal():
    checked()
    if (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Scoring already started")
    files = {}
    for subject, condition, snr in RECORDS:
        name = stub(subject, condition, snr)
        metadata = json.loads((OUTPUT / "predictions/metadata" / f"{name}.json").read_text())
        if metadata["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
            raise ValueError("Prediction used another protocol")
        for method in METHODS:
            path = OUTPUT / "predictions" / method / f"{name}.npz"
            if sha256(path) != metadata["prediction_sha256"][method]:
                raise ValueError("Changed predictions")
            files[str(path.relative_to(OUTPUT))] = sha256(path)
    if len(files) != len(RECORDS) * len(METHODS):
        raise ValueError("Need every prediction before test scoring")
    write_json(
        OUTPUT / "predictions_complete.json",
        dict(completed_at_utc=datetime.now(UTC).isoformat(), files=files),
    )


def _reference_predictions(name):
    """Stage B's sealed predictions, if that run is present and unchanged."""
    manifest = STAGE_B_OUTPUT / "predictions_complete.json"
    if not manifest.exists():
        return {}
    files = json.loads(manifest.read_text())["files"]
    out = {}
    for method in REFERENCE_METHODS:
        relative = f"predictions/{method}/{name}.npz"
        key = next((k for k in files if k.replace("\\", "/") == relative), None)
        path = STAGE_B_OUTPUT / relative
        if key is None or not path.exists() or sha256(path) != files[key]:
            raise ValueError(f"Stage B reference prediction missing or changed: {relative}")
        with np.load(path) as archive:
            out[f"stage_b_{method}"] = (archive["spikes"], archive["sil"])
    return out


@torch.no_grad()
def score(subjects=None):
    checked()
    from adapt_decomp import AdaptationResult
    from adapt_decomp.spikes import get_sil, rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    manifest = OUTPUT / "predictions_complete.json"
    document = json.loads(manifest.read_text())
    for path, digest in document["files"].items():
        if sha256(OUTPUT / path) != digest:
            raise ValueError("Prediction changed")
    marker = OUTPUT / "scoring_started.json"
    if not marker.exists():
        write_json(
            marker,
            dict(
                started_at_utc=datetime.now(UTC).isoformat(),
                prediction_manifest_sha256=sha256(manifest),
            ),
        )
    elif json.loads(marker.read_text())["prediction_manifest_sha256"] != sha256(manifest):
        raise ValueError("Scoring manifest changed")
    for subject, condition, snr in RECORDS:
        if subjects and subject not in subjects:
            continue
        name = stub(subject, condition, snr)
        destination = OUTPUT / "scored" / f"{name}.json"
        if destination.exists():
            continue
        record = prepare(subject, condition, snr)
        baseline = AdaptationResult.load(pareto_path(subject, condition, snr))
        sil_pareto = (
            np.asarray(baseline.sil)
            if baseline.sil is not None
            else get_sil(
                baseline.sources, baseline.spikes, record["config"].spike_min_dist, peak_power=2
            ).numpy()
        )
        predictions = dict(pareto=(baseline.spikes.numpy().astype(np.uint8), sil_pareto))
        predictions.update(_reference_predictions(name))
        for method in METHODS:
            with np.load(OUTPUT / "predictions" / method / f"{name}.npz") as archive:
                predictions[method] = (archive["spikes"], archive["sil"])
        n = len(record["x"])
        truth = load_gt(
            DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
            n_samples=n,
        )[:, record["calibration"].gt_matched_indices]
        with np.load(
            DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
        ) as archive:
            angle = np.asarray(archive["angle"], dtype=float)[:n]
        phases = angle_phases(angle, 2048, rest_deg=ANGLE_REST_DEG, rest_speed=SPEED_REST_DEG_S)
        rows = []
        for method, (binary, sil) in predictions.items():
            roa, _, _ = rate_of_agreement_paired(truth, binary, fs=2048, tol_spike_ms=2)
            if method == "pareto" and not np.allclose(roa, baseline.roa, rtol=0, atol=1e-10):
                raise ValueError("Original baseline scorer changed")
            first_end = (5 if condition == "staircase" else 10) * 2048
            windows = dict(
                first=slice(0, first_end), middle=slice(first_end, -10240), last=slice(-10240, None)
            )
            fixed = {
                phase: rate_of_agreement_paired(
                    truth[window], binary[window], fs=2048, tol_spike_ms=2
                )[0]
                for phase, window in windows.items()
            }
            by_angle = {}
            for phase, mask in phases.items():
                if mask.sum() < 2048:
                    by_angle[phase] = np.full(truth.shape[1], np.nan)
                    continue
                by_angle[phase] = rate_of_agreement_paired(
                    masked(truth, mask), masked(binary, mask), fs=2048, tol_spike_ms=2
                )[0]
            for unit, value in enumerate(roa):
                nr, npred = int(truth[:, unit].sum()), int(binary[:, unit].sum())
                tp = round(float(value) * (nr + npred) / (1 + float(value)))
                quality = float(sil[unit])
                rows.append(
                    dict(
                        subject=subject,
                        condition=condition,
                        snr_db=snr,
                        unit=unit,
                        gt_unit=int(record["calibration"].gt_matched_indices[unit]),
                        method=method,
                        roa=float(value),
                        precision=tp / npred if npred else 0,
                        recall=tp / nr if nr else 0,
                        f1=2 * tp / (nr + npred) if nr + npred else 0,
                        sil=quality if np.isfinite(quality) else None,
                        first_hold_roa=float(fixed["first"][unit]),
                        ramp_roa=float(fixed["middle"][unit]),
                        last_hold_roa=float(fixed["last"][unit]),
                        **{
                            f"angle_{phase}_roa": (
                                float(values[unit]) if np.isfinite(values[unit]) else None
                            )
                            for phase, values in by_angle.items()
                        },
                    )
                )
        write_json(destination, dict(rows=rows))
        print("scored", name, flush=True)
