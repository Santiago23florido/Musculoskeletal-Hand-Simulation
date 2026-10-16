"""Locked out-of-subject scoring after BOTH seeds of ALL folds are frozen."""

import json

import numpy as np
import torch

from benchmarks.stage_a.common import pareto_path
from benchmarks.stage_a.evaluate import write_csv

from .common import CONDITIONS, DATA, OUTPUT, sha256, stub, write_json
from .data import prepare
from .experiment import SPEC, checked_protocol
from .model import ResidualGate
from .replay import replay


def load_models(subject, locked):
    models = []
    for seed in SPEC["seeds"]:
        path = OUTPUT / "folds" / subject / f"seed{seed}" / "selected.pt"
        if sha256(path) != locked["models"][subject][str(seed)]["sha256"]:
            raise ValueError("Selected weights changed after freezing")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = ResidualGate(**checkpoint["spec"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def evaluate(subject):
    from adapt_decomp import AdaptationResult
    from adapt_decomp.spikes import get_sil, rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    checked_protocol()
    locked = json.loads((OUTPUT / "locked.json").read_text())
    if locked["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise ValueError("Protocol changed after freezing")
    models = load_models(subject, locked)
    folder = OUTPUT / "evaluations" / subject
    folder.mkdir(parents=True, exist_ok=True)
    plan = dict(
        subject=subject,
        locked_sha256=sha256(OUTPUT / "locked.json"),
        code_sha256=sha256(__file__),
        methods=["hybrid", "seed0", "seed1", "pareto"],
    )
    if (folder / "plan.json").exists() and json.loads((folder / "plan.json").read_text()) != plan:
        raise ValueError("Evaluation changed; preserve the old experiment")
    write_json(folder / "plan.json", plan)
    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            name = stub(subject, condition, snr)
            destination = folder / "records" / f"{name}.json"
            if destination.exists():
                continue
            record = prepare(subject, condition, snr)
            methods = {"hybrid": models, "seed0": [models[0]], "seed1": [models[1]]}
            predictions, qualities, timings = {}, {}, {}
            # Produce and save every learned prediction before opening GT.
            for method, selected in methods.items():
                result = replay(record, selected, collect=True)
                binary = result.pop("predictions")
                sources = result.pop("sources")
                qualities[method] = get_sil(
                    torch.from_numpy(sources),
                    torch.from_numpy(binary.astype(np.int32)),
                    record["config"].spike_min_dist,
                    peak_power=2,
                ).numpy()
                predictions[method] = binary
                timings[method] = result
                path = folder / "predictions" / method / f"{name}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, spikes=binary)
            baseline = AdaptationResult.load(pareto_path(subject, condition, snr))
            predictions["pareto"] = baseline.spikes.numpy().astype(np.uint8)
            qualities["pareto"] = (
                np.asarray(baseline.sil)
                if baseline.sil is not None
                else get_sil(
                    baseline.sources, baseline.spikes, record["config"].spike_min_dist, peak_power=2
                ).numpy()
            )
            truth = load_gt(
                DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
                n_samples=len(record["base"]),
            )[:, record["calibration"].gt_matched_indices]
            rows = []
            for method, binary in predictions.items():
                roa, _, _ = rate_of_agreement_paired(truth, binary, fs=2048, tol_spike_ms=2)
                if method == "pareto" and not np.allclose(roa, baseline.roa, rtol=0, atol=1e-10):
                    raise ValueError("Pareto scoring no longer matches the original comparator")
                first_end = (5 if condition == "staircase" else 10) * 2048
                phases = {}
                for phase, window in dict(
                    first=slice(0, first_end),
                    middle=slice(first_end, -5 * 2048),
                    last=slice(-5 * 2048, None),
                ).items():
                    phases[phase], _, _ = rate_of_agreement_paired(
                        truth[window], binary[window], fs=2048, tol_spike_ms=2
                    )
                for unit in range(len(roa)):
                    nr, npred = int(truth[:, unit].sum()), int(binary[:, unit].sum())
                    tp = round(float(roa[unit]) * (nr + npred) / (1 + float(roa[unit])))
                    sil = float(qualities[method][unit])
                    rows.append(
                        dict(
                            subject=subject,
                            condition=condition,
                            snr_db=snr,
                            unit=unit,
                            gt_unit=int(record["calibration"].gt_matched_indices[unit]),
                            method=method,
                            roa=float(roa[unit]),
                            precision=tp / npred if npred else 0,
                            recall=tp / nr if nr else 0,
                            f1=2 * tp / (nr + npred) if nr + npred else 0,
                            sil=sil if np.isfinite(sil) else None,
                            first_hold_roa=float(phases["first"][unit]),
                            ramp_roa=float(phases["middle"][unit]),
                            last_hold_roa=float(phases["last"][unit]),
                        )
                    )
            write_json(
                destination,
                dict(
                    rows=rows,
                    timing=timings,
                    prepared_sha256=sha256(OUTPUT / "features" / name / "metadata.json"),
                ),
            )
            print(subject, condition, snr, "scored", flush=True)
    rows = []
    for path in sorted((folder / "records").glob("*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    write_csv(folder / "per_unit.csv", rows)
