"""Freeze all folds, persist all neural predictions, then score outer subjects."""

import json
import time
from datetime import UTC, datetime

import numpy as np
import torch

from benchmarks.stage_b.common import OUTPUT as ORIGINAL_OUTPUT
from benchmarks.stage_b.data import prepare
from benchmarks.stage_b.experiment import load_models as original_models

from .common import CONDITIONS, DATA, OUTPUT, SUBJECTS, pareto_path, sha256, stub, write_json
from .experiment import checked, load_models
from .inference import build_tables, predict

METHODS = dict(aligned_k="k", aligned_b="b", gain_only_b="b", static_mvdr="static")


@torch.no_grad()
def predict_subject(subject):
    if subject not in SUBJECTS[1:] or (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Invalid subject or scoring has started")
    models = load_models(subject)
    old_models = original_models(subject)
    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            name = stub(subject, condition, snr)
            done = OUTPUT / "predictions/metadata" / f"{name}.json"
            if done.exists():
                metadata = json.loads(done.read_text())
                if metadata["locked_sha256"] != sha256(OUTPUT / "locked.json"):
                    raise ValueError("Wrong frozen model")
                for method, digest in metadata["prediction_sha256"].items():
                    if sha256(OUTPUT / "predictions" / method / f"{name}.npz") != digest:
                        raise ValueError("Predictions changed")
                continue
            tick = time.perf_counter()
            record = prepare(subject, condition, snr)
            table_tick = time.perf_counter()
            tables = build_tables(record, models)
            old_tables = build_tables(record, old_models)
            metadata = dict(methods={}, prediction_sha256={})
            metadata["morph_table_calibration_seconds"] = time.perf_counter() - table_tick
            for method, mode in METHODS.items():
                selected_models = old_models if method == "gain_only_b" else models
                result = predict(
                    record,
                    [] if mode == "static" else selected_models,
                    mode=mode,
                    tables=old_tables if method == "gain_only_b" else tables,
                )
                # SIL is well-defined here: each identity has one synthesized scalar source.
                from adapt_decomp.spikes import get_sil

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
                prepared_sha256=sha256(ORIGINAL_OUTPUT / "features" / name / "metadata.json"),
                locked_sha256=sha256(OUTPUT / "locked.json"),
                completed_at_utc=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.perf_counter() - tick,
                outer_true_spikes_loaded=False,
            )
            write_json(done, metadata)
            print("predicted", name, round(metadata["elapsed_seconds"], 2), flush=True)


def seal():
    checked()
    if (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Scoring already started")
    files = {}
    for subject in SUBJECTS[1:]:
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                name = stub(subject, condition, snr)
                metadata = json.loads(
                    (OUTPUT / "predictions/metadata" / f"{name}.json").read_text()
                )
                if metadata["locked_sha256"] != sha256(OUTPUT / "locked.json"):
                    raise ValueError("Prediction used another freeze")
                for method in METHODS:
                    path = OUTPUT / "predictions" / method / f"{name}.npz"
                    if sha256(path) != metadata["prediction_sha256"][method]:
                        raise ValueError("Changed predictions")
                    files[str(path.relative_to(OUTPUT))] = sha256(path)
    if len(files) != 320:
        raise ValueError("Need every Stage B prediction before test scoring")
    write_json(
        OUTPUT / "predictions_complete.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            files=files,
            locked_sha256=sha256(OUTPUT / "locked.json"),
        ),
    )


def start_scoring():
    checked()
    manifest = OUTPUT / "predictions_complete.json"
    document = json.loads(manifest.read_text())
    if len(document["files"]) != 320:
        raise ValueError("Incomplete predictions")
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


@torch.no_grad()
def score(subject=None):
    checked()
    from adapt_decomp import AdaptationResult
    from adapt_decomp.spikes import rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    if subject is not None and subject not in SUBJECTS[1:]:
        raise ValueError("Unknown scoring partition")
    marker = OUTPUT / "scoring_started.json"
    if not marker.exists():
        raise ValueError("Validate the full manifest with start-scoring before partition scoring")
    if json.loads(marker.read_text())["prediction_manifest_sha256"] != sha256(
        OUTPUT / "predictions_complete.json"
    ):
        raise ValueError("Scoring manifest changed")
    requested = [subject] if subject else SUBJECTS[1:]
    for subject in requested:
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                name = stub(subject, condition, snr)
                destination = OUTPUT / "scored" / f"{name}.json"
                if destination.exists():
                    continue
                record = prepare(subject, condition, snr)
                baseline = AdaptationResult.load(pareto_path(subject, condition, snr))
                binary = dict(pareto=baseline.spikes.numpy().astype(np.uint8))
                if baseline.sil is None:
                    from adapt_decomp.spikes import get_sil

                    baseline_sil = get_sil(
                        baseline.sources,
                        baseline.spikes,
                        record["config"].spike_min_dist,
                        peak_power=2,
                    ).numpy()
                else:
                    baseline_sil = np.asarray(baseline.sil)
                sil = dict(pareto=baseline_sil)
                state_errors, statistics = {}, {}
                for method in METHODS:
                    with np.load(OUTPUT / "predictions" / method / f"{name}.npz") as archive:
                        binary[method] = archive["spikes"]
                        sil[method] = archive["sil"]
                        if method in ("aligned_b", "gain_only_b"):
                            after_calibration = slice(51, None)
                            state_errors[method] = dict(
                                angle_rmse_degrees=float(
                                    np.sqrt(
                                        np.mean(
                                            (
                                                archive["q"][after_calibration]
                                                - record["q"].numpy()[after_calibration]
                                            )
                                            ** 2
                                        )
                                    )
                                    * 40
                                )
                            )
                truth = load_gt(
                    DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
                    n_samples=len(record["x"]),
                )[:, record["calibration"].gt_matched_indices]
                rows = []
                for method, prediction in binary.items():
                    roa, _, _ = rate_of_agreement_paired(truth, prediction, fs=2048, tol_spike_ms=2)
                    if method == "pareto" and not np.allclose(
                        roa, baseline.roa, rtol=0, atol=1e-10
                    ):
                        raise ValueError("Original baseline scorer changed")
                    first_end = (5 if condition == "staircase" else 10) * 2048
                    phases = {}
                    for phase, window in dict(
                        first=slice(0, first_end),
                        middle=slice(first_end, -10240),
                        last=slice(-10240, None),
                    ).items():
                        phases[phase], _, _ = rate_of_agreement_paired(
                            truth[window], prediction[window], fs=2048, tol_spike_ms=2
                        )
                    for unit, value in enumerate(roa):
                        nr, npred = int(truth[:, unit].sum()), int(prediction[:, unit].sum())
                        tp = round(float(value) * (nr + npred) / (1 + float(value)))
                        quality = float(sil[method][unit])
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
                                first_hold_roa=float(phases["first"][unit]),
                                ramp_roa=float(phases["middle"][unit]),
                                last_hold_roa=float(phases["last"][unit]),
                            )
                        )
                original_rows = json.loads(
                    (ORIGINAL_OUTPUT / "scored" / f"{name}.json").read_text()
                )["rows"]
                rows.extend(
                    dict(r, method="original_b") for r in original_rows if r["method"] == "moca_b"
                )
                with np.load(ORIGINAL_OUTPUT / "predictions/moca_b" / f"{name}.npz") as original:
                    with np.load(OUTPUT / "predictions/aligned_b" / f"{name}.npz") as aligned:
                        if not np.array_equal(original["q"], aligned["q"]):
                            raise ValueError("The frozen observer changed")
                write_json(destination, dict(rows=rows, state=state_errors, statistics=statistics))
                print("scored", name, flush=True)
