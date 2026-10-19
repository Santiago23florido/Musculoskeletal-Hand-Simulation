import json
import time
from datetime import UTC, datetime

import numpy as np

from benchmarks.stage_b.common import OUTPUT as STAGE_B_OUTPUT
from benchmarks.stage_b.data import prepare as calibration_record

from .common import CONDITIONS, DATA, OUTPUT, SUBJECTS, sha256, stub, write_json
from .data import decode, prepare
from .experiment import checked, load_models, logits


def predict_subject(subject):
    if subject not in SUBJECTS[1:] or (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Invalid or already scored prediction")
    models, selection = load_models(subject)
    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            name = stub(subject, condition, snr)
            done = OUTPUT / "predictions/metadata" / f"{name}.json"
            if done.exists():
                continue
            r = prepare(subject, condition, snr)
            tick = time.perf_counter()
            output = np.mean([logits(m, r, "cpu") for m in models], axis=0)
            prior = (r["primary"].astype(np.float32) * 2 - 1) * 2.1972245773362196
            probability = 1 / (1 + np.exp(-(prior + selection["strength"] * (output - prior))))
            primary = decode(r, probability, selection["threshold"])
            baseline = decode(r, np.where(r["primary"], 0.9, 0.1).astype(np.float32), 0.5)
            methods = dict(posterior=primary, rescue_union=np.maximum(primary, baseline))
            hashes = {}
            for method, binary in methods.items():
                path = OUTPUT / "predictions" / method / f"{name}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, spikes=binary)
                hashes[method] = sha256(path)
            write_json(
                done,
                dict(
                    completed_at_utc=datetime.now(UTC).isoformat(),
                    prediction_sha256=hashes,
                    locked_sha256=sha256(OUTPUT / "locked.json"),
                    feature_metadata_sha256=sha256(OUTPUT / "features" / name / "metadata.json"),
                    outer_true_spikes_loaded=False,
                    elapsed_seconds=time.perf_counter() - tick,
                    scope=(
                        "cached Pareto/fixed sources; excludes upstream decomposition "
                        "and disk preparation"
                    ),
                ),
            )
            print("predicted", name, flush=True)


def seal():
    checked()
    files = {}
    for subject in SUBJECTS[1:]:
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                name = stub(subject, condition, snr)
                metadata = json.loads(
                    (OUTPUT / "predictions/metadata" / f"{name}.json").read_text()
                )
                if metadata["locked_sha256"] != sha256(OUTPUT / "locked.json"):
                    raise ValueError("Wrong frozen predictions")
                for method, digest in metadata["prediction_sha256"].items():
                    path = OUTPUT / "predictions" / method / f"{name}.npz"
                    if sha256(path) != digest:
                        raise ValueError("Prediction changed")
                    files[str(path.relative_to(OUTPUT))] = digest
    if len(files) != 160:
        raise ValueError("Incomplete predictions")
    write_json(
        OUTPUT / "predictions_complete.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            locked_sha256=sha256(OUTPUT / "locked.json"),
            files=files,
        ),
    )


def start_scoring():
    checked()
    path = OUTPUT / "predictions_complete.json"
    manifest = json.loads(path.read_text())
    if len(manifest["files"]) != 160:
        raise ValueError("Incomplete predictions")
    for name, digest in manifest["files"].items():
        if sha256(OUTPUT / name) != digest:
            raise ValueError("Predictions changed")
    marker = OUTPUT / "scoring_started.json"
    if marker.exists():
        raise ValueError("Scoring already started")
    write_json(
        marker, dict(started_at_utc=datetime.now(UTC).isoformat(), manifest_sha256=sha256(path))
    )


def score(subject):
    checked()
    from adapt_decomp.spikes import rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    if subject not in SUBJECTS[1:]:
        raise ValueError("Unknown scoring subject")
    marker = json.loads((OUTPUT / "scoring_started.json").read_text())
    if marker["manifest_sha256"] != sha256(OUTPUT / "predictions_complete.json"):
        raise ValueError("Scoring manifest changed")
    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            name = stub(subject, condition, snr)
            destination = OUTPUT / "scored" / f"{name}.json"
            if destination.exists():
                continue
            r = calibration_record(subject, condition, snr)
            truth = load_gt(
                DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
                n_samples=len(r["x"]),
            )[:, r["calibration"].gt_matched_indices]
            original = json.loads((STAGE_B_OUTPUT / "scored" / f"{name}.json").read_text())["rows"]
            rows = [
                dict(row, method="previous_b" if row["method"] == "moca_b" else "pareto")
                for row in original
                if row["method"] in ("pareto", "moca_b")
            ]
            first_end = (5 if condition == "staircase" else 10) * 2048
            for method in ("posterior", "rescue_union"):
                with np.load(OUTPUT / "predictions" / method / f"{name}.npz") as a:
                    binary = a["spikes"]
                roa, _, _ = rate_of_agreement_paired(truth, binary, fs=2048, tol_spike_ms=2)
                phases = {}
                for phase, window in dict(
                    first=slice(0, first_end),
                    middle=slice(first_end, -10240),
                    last=slice(-10240, None),
                ).items():
                    phases[phase], _, _ = rate_of_agreement_paired(
                        truth[window], binary[window], fs=2048, tol_spike_ms=2
                    )
                for j, value in enumerate(roa):
                    nr, npred = int(truth[:, j].sum()), int(binary[:, j].sum())
                    tp = round(float(value) * (nr + npred) / (1 + float(value)))
                    rows.append(
                        dict(
                            subject=subject,
                            condition=condition,
                            snr_db=snr,
                            unit=j,
                            gt_unit=int(r["calibration"].gt_matched_indices[j]),
                            method=method,
                            roa=float(value),
                            precision=tp / npred if npred else 0,
                            recall=tp / nr if nr else 0,
                            f1=2 * tp / (nr + npred) if nr + npred else 0,
                            sil=None,
                            first_hold_roa=float(phases["first"][j]),
                            ramp_roa=float(phases["middle"][j]),
                            last_hold_roa=float(phases["last"][j]),
                        )
                    )
            write_json(destination, dict(rows=rows))
            print("scored", name, flush=True)
