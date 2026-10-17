"""Post-hoc training/validation diagnostics; never retrains or selects models."""

import json

import numpy as np
import torch

from benchmarks.stage_b.common import DATA, OUTPUT, ROOT, reference, split, write_json
from benchmarks.stage_b.data import prepare, supervision
from benchmarks.stage_b.experiment import load_models
from benchmarks.stage_b.inference import build_tables, extended_template, predict


def mean_cosine(a, b):
    a, b = a.flatten(1), b.flatten(1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    reference()
    from adapt_decomp.spikes import rate_of_agreement_paired
    from adapt_decomp.utils import load_gt

    outer = "sub-02"
    models = load_models(outer)
    results = []
    for subject, snr, phase in (
        ("sub-01", 30, "training"),
        (split(outer)["validation"][0], 25, "validation"),
    ):
        condition = "staircase"
        record = prepare(subject, condition, snr)
        labels = supervision(subject, condition, snr, outer=outer, phase=phase)
        cal = labels["cal"][labels["unit"]]
        predicted = []
        for start in range(0, len(cal), 16):
            batch = slice(start, start + 16)
            predicted.append(
                torch.stack([m(cal[batch], labels["q"][batch]) for m, _ in models]).mean(0)
            )
        predicted = torch.cat(predicted)
        amplitude = record["amplitude"][labels["unit"]]
        projected_target = extended_template(labels["target"], record["projection"], amplitude)
        projected_prediction = extended_template(predicted, record["projection"], amplitude)
        projected_static = extended_template(cal, record["projection"], amplitude)

        def nmse(a, b):
            return float(((a - b).flatten(1).square().sum(1) / b.flatten(1).square().sum(1)).mean())

        row = dict(
            subject=subject,
            condition=condition,
            snr_db=snr,
            phase=phase,
            pairs=len(cal),
            learned_waveform_nmse=nmse(predicted, labels["target"]),
            static_waveform_nmse=nmse(cal, labels["target"]),
            learned_projected_nmse=nmse(projected_prediction, projected_target),
            static_projected_nmse=nmse(projected_static, projected_target),
            learned_projected_cosine=mean_cosine(projected_prediction, projected_target),
            static_projected_cosine=mean_cosine(projected_static, projected_target),
            roa_pct={},
        )
        tables = build_tables(record, models)
        truth = load_gt(
            DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
            n_samples=len(record["x"]),
        )[:, record["calibration"].gt_matched_indices]
        for mode in ("k", "b", "static"):
            result = predict(record, [] if mode == "static" else models, mode=mode, tables=tables)
            roa, _, _ = rate_of_agreement_paired(truth, result["spikes"], fs=2048, tol_spike_ms=2)
            row["roa_pct"][mode] = float(np.mean(roa) * 100)
        print(json.dumps(row), flush=True)
        results.append(row)
    result = dict(
        scope="post-hoc two-record diagnostic of frozen outer-sub-02 models; no tuning",
        targets="estimated clean-EMG GT-triggered averages; not original generator MUAPs",
        interpretation=(
            "Waveform error and projected central-template error weight different directions. "
            "These two records diagnose a mismatch; they do not establish its unique cause."
        ),
        records=results,
    )
    write_json(OUTPUT / "report/training_diagnostic.json", result)
    write_json(ROOT / "docs/benchmarks/stage_b_training_diagnostic.json", result)


if __name__ == "__main__":
    main()
