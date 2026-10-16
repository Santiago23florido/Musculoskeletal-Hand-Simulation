"""Locked P1 evaluation. Spike truth enters only after predictions are produced."""

import csv
import json
import shutil
import time
from datetime import UTC, datetime

import numpy as np
import torch
from scipy.stats import wilcoxon

from .common import CONDITIONS, DATA, OUTPUT, POOL, ROOT, pareto_path, sha256, stub, write_json
from .data import prepare
from .inference import predict
from .model import MotionCorrection


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MotionCorrection(**saved["spec"])
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def summarize(rows):
    summaries = {}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        scores = np.array([r["roa"] for r in selected])
        summaries[method] = dict(
            recordings=len({(r["subject"], r["condition"], r["snr_db"]) for r in selected}),
            unit_instances=len(scores),
            mean_roa_pct=float(scores.mean() * 100),
            median_roa_pct=float(np.median(scores) * 100),
            pct_units_roa_ge_90=float(np.mean(scores >= 0.9) * 100),
            by_snr={
                str(s): float(np.mean([r["roa"] for r in selected if r["snr_db"] == s]) * 100)
                for s in (15, 20, 25, 30)
                if any(r["snr_db"] == s for r in selected)
            },
            by_condition={
                c: float(np.mean([r["roa"] for r in selected if r["condition"] == c]) * 100)
                for c in CONDITIONS
                if any(r["condition"] == c for r in selected)
            },
            trained_subject_mean_roa_pct=float(
                np.mean([r["roa"] for r in selected if r["subject"] == "sub-01"]) * 100
            )
            if any(r["subject"] == "sub-01" for r in selected)
            else None,
            unseen_subject_mean_roa_pct=float(
                np.mean([r["roa"] for r in selected if r["subject"] != "sub-01"]) * 100
            )
            if any(r["subject"] != "sub-01" for r in selected)
            else None,
        )
    return summaries


def gates(summaries, method="stage_a"):
    actual = summaries[method]["by_condition"]
    baseline = summaries["pareto"]["by_condition"]
    result = {}
    for condition, minimum in [
        ("triangular-ramp5s", 5),
        ("triangular-ramp10s", 5),
        ("staircase", 0),
    ]:
        if condition in actual:
            delta = actual[condition] - baseline[condition]
            result[condition] = dict(
                mean_roa_pct=actual[condition],
                pareto_roa_pct=baseline[condition],
                delta_pp=delta,
                required_delta_pp=minimum,
                passed=delta >= minimum,
            )
    return result


def paired_statistics(rows, method="stage_a", seed=2026):
    truth = {
        (r["subject"], r["condition"], r["snr_db"], r["unit"]): r["roa"]
        for r in rows
        if r["method"] == "pareto"
    }
    rng = np.random.default_rng(seed)
    results = []
    for condition in ("staircase", "triangular-ramp10s", "triangular-ramp5s"):
        values = [r for r in rows if r["method"] == method and r["condition"] == condition]
        if not values:
            continue
        differences = (
            np.array(
                [
                    r["roa"] - truth[(r["subject"], r["condition"], r["snr_db"], r["unit"])]
                    for r in values
                ]
            )
            * 100
        )
        p = 1.0 if np.all(differences == 0) else float(wilcoxon(differences).pvalue)
        # Resample subjects as clusters: unit-instances within a subject are dependent.
        subjects = sorted({r["subject"] for r in values})
        clusters = [differences[[r["subject"] == s for r in values]] for s in subjects]
        boot = [
            float(
                np.concatenate(
                    [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
                ).mean()
            )
            for _ in range(2000)
        ]
        results.append(
            dict(
                condition=condition,
                mean_delta_pp=float(differences.mean()),
                wilcoxon_unit_p=p,
                subject_cluster_bootstrap_95_ci_pp=np.quantile(boot, [0.025, 0.975]).tolist(),
            )
        )
    order = sorted(range(len(results)), key=lambda i: results[i]["wilcoxon_unit_p"])
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * results[index]["wilcoxon_unit_p"]))
        results[index]["wilcoxon_holm_p"] = previous
    return results


def evaluate(checkpoint, name, *, subjects=None, conditions=None, snrs=None):
    from adapt_decomp import AdaptationResult
    from adapt_decomp.spikes import get_sil, rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    output = OUTPUT / "evaluations" / name
    output.mkdir(parents=True, exist_ok=True)
    model, saved = load_model(checkpoint)
    subjects = subjects or [f"sub-0{i}" for i in range(1, 6)]
    conditions, snrs = conditions or CONDITIONS, snrs or [30, 25, 20, 15]
    methods = {
        "stage_a": dict(coupling=True),
        "stage_a_no_coupling": dict(coupling=False),
        "identity_sv_sd": dict(coupling=True),
    }
    plan = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        epoch=saved["epoch"],
        selection_loss=saved["selection_loss"],
        subjects=subjects,
        conditions=conditions,
        snrs=snrs,
        methods=methods,
        checkpoint_selection_used_evaluation_truth=False,
        primary_method="stage_a",
        threads=torch.get_num_threads(),
        ablations=list(methods)[1:],
        protocol="P1: full recording, 2ms tolerance, authors' global lag +/-40ms",
        inference_code_sha256=sha256(ROOT / "benchmarks/stage_a/inference.py"),
        evaluator_code_sha256=sha256(__file__),
    )
    if (output / "plan.json").exists():
        old = json.loads((output / "plan.json").read_text())
        if old != plan:
            raise ValueError("Evaluation plan changed; use a new name")
    write_json(output / "plan.json", plan)
    snapshot = output / "source"
    snapshot.mkdir(exist_ok=True)
    for source in (ROOT / "benchmarks/stage_a").glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    started, rows, timings = time.perf_counter(), [], []
    for number, (subject, condition, snr) in enumerate(
        ((s, c, n) for s in subjects for c in conditions for n in snrs), 1
    ):
        name_stub = stub(subject, condition, snr)
        done = output / "records" / f"{name_stub}.json"
        if done.exists():
            record_result = json.loads(done.read_text())
            rows.extend(record_result["rows"])
            timings.extend(record_result["timings"])
            continue
        record = prepare(
            subject, condition, snr, persist=subject == "sub-01" and snr == 30 and condition in POOL
        )
        # Predictions are generated and persisted BEFORE this function opens GT.
        predictions = {}
        record_times = []
        for method, options in methods.items():
            binary, speed, sources = predict(
                record, None if method == "identity_sv_sd" else model, **options
            )
            sil = get_sil(
                torch.from_numpy(sources),
                torch.from_numpy(binary.astype(np.int32)),
                record["config"].spike_min_dist,
                peak_power=2,
            ).numpy()
            predictions[method] = (binary, sil)
            speed.update(subject=subject, condition=condition, snr_db=snr, method=method)
            record_times.append(speed)
            destination = output / "predictions" / method / f"{name_stub}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                spikes=binary,
                ground_truth_indices=record["calibration"].gt_matched_indices,
            )
            del sources
        indices = record["calibration"].gt_matched_indices
        gt_path = DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz"
        truth = load_gt(gt_path, n_samples=len(record["z"]))[:, indices]
        baseline = AdaptationResult.load(pareto_path(subject, condition, snr))
        baseline_sil = baseline.sil
        if baseline_sil is None:
            baseline_sil = get_sil(
                baseline.sources, baseline.spikes, record["config"].spike_min_dist, peak_power=2
            ).numpy()
        predictions["pareto"] = (baseline.spikes.numpy().astype(np.uint8), np.asarray(baseline_sil))
        record_rows = []
        for method, (binary, sil) in predictions.items():
            roa, _, lags = rate_of_agreement_paired(truth, binary, fs=2048, tol_spike_ms=2)
            if method == "pareto" and not np.allclose(roa, baseline.roa, rtol=0, atol=1e-10):
                raise ValueError("Recomputed Pareto comparator disagrees with published scores")
            # Phase metrics use the same evaluator, with independent lag per phase,
            # just as the authors' fdsi_common.compute_roa_subset does.
            last, _, _ = rate_of_agreement_paired(
                truth[-5 * 2048 :], binary[-5 * 2048 :], fs=2048, tol_spike_ms=2
            )
            first_end = (10 if condition.startswith("triangular") else 5) * 2048
            first, _, _ = rate_of_agreement_paired(
                truth[:first_end], binary[:first_end], fs=2048, tol_spike_ms=2
            )
            middle, _, _ = rate_of_agreement_paired(
                truth[first_end : -5 * 2048],
                binary[first_end : -5 * 2048],
                fs=2048,
                tol_spike_ms=2,
            )
            for unit in range(len(indices)):
                ref = np.flatnonzero(truth[:, unit])
                pred = np.flatnonzero(binary[:, unit])
                # RoA=TP/(Nref+Npred-TP), so counts follow algebraically from
                # the exact reference score, including shifted edge detections.
                tp = round(float(roa[unit]) * (len(ref) + len(pred)) / (1 + float(roa[unit])))
                record_rows.append(
                    dict(
                        subject=subject,
                        condition=condition,
                        snr_db=snr,
                        method=method,
                        unit=unit,
                        gt_unit=int(indices[unit]),
                        roa=float(roa[unit]),
                        precision=tp / len(pred) if len(pred) else 0,
                        recall=tp / len(ref) if len(ref) else 0,
                        f1=2 * tp / (len(ref) + len(pred)) if len(ref) + len(pred) else 0,
                        sil=float(sil[unit]) if np.isfinite(sil[unit]) else None,
                        first_hold_roa=float(first[unit]),
                        ramp_roa=float(middle[unit]),
                        last_hold_roa=float(last[unit]),
                    )
                )
        write_json(
            done,
            dict(
                rows=record_rows,
                timings=record_times,
                calibration_sha256=record["metadata"]["calibration_sha256"],
            ),
        )
        rows.extend(record_rows)
        timings.extend(record_times)
        write_csv(output / "per_unit.csv", rows)
        total = len(subjects) * len(conditions) * len(snrs)
        print(f"{number}/{total} {name_stub}: predictions scored", flush=True)
        del record
    return finalize_results(rows, timings, plan, output, time.perf_counter() - started)


def finalize_results(rows, timings, plan, output, elapsed_seconds):
    subjects, conditions, snrs = plan["subjects"], plan["conditions"], plan["snrs"]
    summaries = summarize(rows)
    checks = gates(summaries)
    complete = (
        set(subjects) == {f"sub-0{i}" for i in range(1, 6)}
        and set(conditions) == set(CONDITIONS)
        and set(snrs) == {15, 20, 25, 30}
        and len(subjects) * len(conditions) * len(snrs) == 100
    )
    if complete:
        for method in summaries:
            selected = [r for r in rows if r["method"] == method]
            if (
                len(selected) != 768
                or len({(r["subject"], r["condition"], r["snr_db"], r["unit"]) for r in selected})
                != 768
            ):
                raise ValueError("Incomplete or duplicate P1 unit rows")
    summary = dict(
        status="go"
        if complete and all(c["passed"] for c in checks.values())
        else "no-go"
        if complete
        else "partial diagnostic; not a P1 gate",
        executed_at_utc=datetime.now(UTC).isoformat(),
        plan=plan,
        complete_p1=complete,
        summaries=summaries,
        go_no_go=checks,
        statistics=paired_statistics(rows),
        elapsed_seconds=elapsed_seconds,
        timing_scope="offline W0 projection and original detector/SV loops; not live latency",
    )
    for method in summaries:
        values = [r for r in rows if r["method"] == method]
        summary["summaries"][method]["mean_precision"] = float(
            np.mean([r["precision"] for r in values])
        )
        summary["summaries"][method]["mean_recall"] = float(np.mean([r["recall"] for r in values]))
        summary["summaries"][method]["mean_f1"] = float(np.mean([r["f1"] for r in values]))
        summary["summaries"][method]["end_units_roa_ge_90"] = sum(
            r["last_hold_roa"] >= 0.9 for r in values
        )
        finite_sil = [r["sil"] for r in values if r["sil"] is not None]
        summary["summaries"][method]["mean_sil"] = (
            float(np.mean(finite_sil)) if finite_sil else None
        )
        for phase in ("first_hold_roa", "ramp_roa", "last_hold_roa"):
            summary["summaries"][method]["mean_" + phase] = float(
                np.mean([r[phase] for r in values])
            )
        summary["summaries"][method]["recall_threshold_curve"] = [
            dict(threshold=float(t), fraction=float(np.mean([r["recall"] >= t for r in values])))
            for t in np.linspace(0, 1, 21)
        ]
    write_json(output / "summary.json", summary)
    write_csv(output / "timing.csv", timings)
    print(json.dumps({"status": summary["status"], "go_no_go": checks}, indent=2))
    return summary
