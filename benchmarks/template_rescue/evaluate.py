"""Persist ALL predictions before any new true-spike scoring."""

import json
import time
from datetime import UTC, datetime

import numpy as np
import torch

from benchmarks.pareto_residual.evaluate import load_models
from benchmarks.pareto_residual.replay import replay as previous_replay
from benchmarks.stage_a.common import pareto_path

from .common import CONDITIONS, DATA, OUTPUT, PREVIOUS, SUBJECTS, sha256, stub, write_json
from .data import prepare
from .experiment import SPEC, ConstantGate, frozen
from .replay import physical_sources, replay


@torch.no_grad()
def predict(subject_only=None):
    locked = frozen()
    if (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Predictions cannot be regenerated after scoring starts")
    previous = json.loads((PREVIOUS / "locked.json").read_text())
    if subject_only is not None and subject_only not in SUBJECTS[1:]:
        raise ValueError("Unknown outer subject")
    for subject in [subject_only] if subject_only else SUBJECTS[1:]:
        models = load_models(subject, previous)
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                name = stub(subject, condition, snr)
                done = OUTPUT / "predictions" / "metadata" / f"{name}.json"
                if done.exists():
                    metadata = json.loads(done.read_text())
                    if metadata["locked_sha256"] != sha256(OUTPUT / "locked.json"):
                        raise ValueError("Predictions belong to different frozen models")
                    for method, digest in metadata["prediction_sha256"].items():
                        if digest != sha256(OUTPUT / "predictions" / method / f"{name}.npz"):
                            raise ValueError("Predictions changed")
                    continue
                tick = time.perf_counter()
                record = prepare(subject, condition, snr)
                physics = physical_sources(record)
                metadata = dict(physical=physics["diagnostics"], methods={})
                constant = previous_replay(
                    record, [ConstantGate(locked["selections"][subject]["value"])], collect=True
                )
                predictions = {"constant": constant.pop("predictions")}
                constant.pop("sources")
                metadata["methods"]["constant"] = constant
                for method in ("protected", "physical", "rescue"):
                    predictions[method], metrics = replay(record, models, physics, method=method)
                    metadata["methods"][method] = metrics
                hashes = {}
                for method, binary in predictions.items():
                    path = OUTPUT / "predictions" / method / f"{name}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, spikes=binary)
                    hashes[method] = sha256(path)
                metadata.update(
                    locked_sha256=sha256(OUTPUT / "locked.json"),
                    prediction_sha256=hashes,
                    completed_at_utc=datetime.now(UTC).isoformat(),
                    elapsed_seconds=time.perf_counter() - tick,
                    labels_loaded=False,
                    inputs=dict(
                        previous_preparation=record["metadata"],
                        raw_signal_sha256=sha256(
                            DATA / "data" / subject / "noisy" / f"{name}_emg.npz"
                        ),
                        angle_sha256=sha256(
                            DATA
                            / "data"
                            / subject
                            / "clean"
                            / f"{subject}_FDSI_{condition}_angle.npz"
                        ),
                        previous_prediction_sha256=sha256(
                            PREVIOUS
                            / "evaluations"
                            / subject
                            / "predictions"
                            / "hybrid"
                            / f"{name}.npz"
                        ),
                    ),
                )
                write_json(done, metadata)
                print(
                    "predicted",
                    name,
                    round(metadata["elapsed_seconds"], 2),
                    "rescued",
                    metadata["methods"]["rescue"]["recovered_candidate_events"],
                    flush=True,
                )
    if subject_only:
        return
    seal_predictions()


def seal_predictions():
    frozen()
    if (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Scoring has already started")
    for subject in SUBJECTS[1:]:
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                name = stub(subject, condition, snr)
                metadata = json.loads(
                    (OUTPUT / "predictions" / "metadata" / f"{name}.json").read_text()
                )
                if metadata["locked_sha256"] != sha256(OUTPUT / "locked.json"):
                    raise ValueError("Prediction freeze differs")
                for method, digest in metadata["prediction_sha256"].items():
                    if sha256(OUTPUT / "predictions" / method / f"{name}.npz") != digest:
                        raise ValueError("Prediction changed")
    write_json(
        OUTPUT / "predictions_complete.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            records=80,
            locked_sha256=sha256(OUTPUT / "locked.json"),
            prediction_files={
                str(p.relative_to(OUTPUT)): sha256(p)
                for p in sorted((OUTPUT / "predictions").glob("*/*.npz"))
            },
        ),
    )


@torch.no_grad()
def score():
    from adapt_decomp import AdaptationResult
    from adapt_decomp.spikes import rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    frozen()
    manifest_path = OUTPUT / "predictions_complete.json"
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["prediction_files"]) != 320:
        raise ValueError("All 320 new prediction files must exist before opening GT")
    for relative, digest in manifest["prediction_files"].items():
        if sha256(OUTPUT / relative) != digest:
            raise ValueError("Prediction changed before scoring")
    marker = OUTPUT / "scoring_started.json"
    if not marker.exists():
        write_json(
            marker,
            dict(
                started_at_utc=datetime.now(UTC).isoformat(),
                predictions_manifest_sha256=sha256(manifest_path),
            ),
        )
    elif json.loads(marker.read_text())["predictions_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Scoring manifest changed")
    for subject in SUBJECTS[1:]:
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                name = stub(subject, condition, snr)
                done = OUTPUT / "scored" / f"{name}.json"
                if done.exists():
                    continue
                baseline = AdaptationResult.load(pareto_path(subject, condition, snr))
                from benchmarks.stage_a.data import load_calibration

                calibration, config = load_calibration(subject, condition, snr)
                metadata = json.loads(
                    (OUTPUT / "predictions" / "metadata" / f"{name}.json").read_text()
                )
                previous_path = (
                    PREVIOUS / "evaluations" / subject / "predictions" / "hybrid" / f"{name}.npz"
                )
                if sha256(previous_path) != metadata["inputs"]["previous_prediction_sha256"]:
                    raise ValueError("Previous comparator prediction changed")
                predictions = {"pareto": baseline.spikes.numpy().astype(np.uint8)}
                for method in SPEC["methods"][1:]:
                    path = (
                        (
                            PREVIOUS
                            / "evaluations"
                            / subject
                            / "predictions"
                            / "hybrid"
                            / f"{name}.npz"
                        )
                        if method == "previous"
                        else (OUTPUT / "predictions" / method / f"{name}.npz")
                    )
                    with np.load(path) as archive:
                        predictions[method] = archive["spikes"]
                truth = load_gt(
                    DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
                    n_samples=len(baseline.spikes),
                )[:, calibration.gt_matched_indices]
                rows = []
                for method, binary in predictions.items():
                    roa, _, _ = rate_of_agreement_paired(
                        truth, binary, fs=config.fs, tol_spike_ms=2
                    )
                    if method == "pareto" and not np.allclose(
                        roa, baseline.roa, atol=1e-10, rtol=0
                    ):
                        raise ValueError("Baseline scorer changed")
                    first_end = (5 if condition == "staircase" else 10) * config.fs
                    phases = {}
                    for phase, window in dict(
                        first=slice(0, first_end),
                        middle=slice(first_end, -5 * config.fs),
                        last=slice(-5 * config.fs, None),
                    ).items():
                        phases[phase], _, _ = rate_of_agreement_paired(
                            truth[window], binary[window], fs=config.fs, tol_spike_ms=2
                        )
                    for unit, value in enumerate(roa):
                        nr, npred = int(truth[:, unit].sum()), int(binary[:, unit].sum())
                        tp = round(float(value) * (nr + npred) / (1 + float(value)))
                        rows.append(
                            dict(
                                subject=subject,
                                condition=condition,
                                snr_db=snr,
                                unit=unit,
                                gt_unit=int(calibration.gt_matched_indices[unit]),
                                method=method,
                                roa=float(value),
                                precision=tp / npred if npred else 0,
                                recall=tp / nr if nr else 0,
                                f1=2 * tp / (nr + npred) if nr + npred else 0,
                                first_hold_roa=float(phases["first"][unit]),
                                ramp_roa=float(phases["middle"][unit]),
                                last_hold_roa=float(phases["last"][unit]),
                            )
                        )
                write_json(done, dict(rows=rows))
                print("scored", name, flush=True)
