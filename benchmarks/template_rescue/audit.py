"""Perturb future preprocessed batches without opening spike truth."""

import json

import torch

from benchmarks.pareto_residual.evaluate import load_models

from .common import OUTPUT, PREVIOUS, ROOT, write_json
from .data import prepare
from .replay import physical_sources, replay


@torch.no_grad()
def audit():
    record = prepare("sub-01", "triangular-ramp10s", 30)
    batch = record["config"].batch_size
    limit, boundary = 200 * batch, 120 * batch
    for key in ("base", "frozen", "baseline_spikes", "z"):
        record[key] = record[key][:limit].clone()
    record["angle"] = record["angle"][:limit].copy()
    record["features"] = record["features"][:200].clone()
    original = physical_sources(record)
    changed = dict(record)
    for key in ("base", "frozen", "z"):
        changed[key] = record[key].clone()
        changed[key][boundary:] *= -2
    changed["angle"] = record["angle"].copy()
    changed["angle"][boundary:] += 13
    changed["features"] = record["features"].clone()
    changed["features"][120:] *= -2
    # Changing base sources requires recomputing their own independent detector
    # outputs in the suffix. The unchanged prefix is the audited part.
    from benchmarks.pareto_residual.detector import Detector

    detector = Detector(record["calibration"], record["config"])
    changed["baseline_spikes"] = record["baseline_spikes"].clone()
    for index, start in enumerate(range(0, limit, batch)):
        trim = record["config"].ext_fact if index == 0 else 0
        binary, _ = detector.step(changed["base"][start + trim : start + batch], index)
        changed["baseline_spikes"][start + trim : start + batch] = binary
    alternative = physical_sources(changed)
    checked = {}
    for key in ("delta", "template_sources", "evidence", "candidates"):
        checked[key] = bool(torch.equal(original[key][:boundary], alternative[key][:boundary]))
    models = load_models("sub-02", json.loads((PREVIOUS / "locked.json").read_text()))
    a, _ = replay(record, models, original, method="rescue")
    b, _ = replay(changed, models, alternative, method="rescue")
    checked["rescued_spikes"] = bool((a[:boundary] == b[:boundary]).all())
    if not all(checked.values()):
        raise ValueError(f"Future perturbation changed a prefix: {checked}")
    result = dict(
        record="sub-01/triangular-ramp10s/30dB",
        boundary_samples=boundary,
        unchanged_prefix_checks=checked,
        scope="100ms batch causality AFTER supplied offline preprocessing and initial calibration",
        labels_loaded=False,
    )
    write_json(OUTPUT / "causality_audit.json", result)
    write_json(ROOT / "docs/benchmarks/template_rescue_causality.json", result)
    print(json.dumps(result, indent=2))
