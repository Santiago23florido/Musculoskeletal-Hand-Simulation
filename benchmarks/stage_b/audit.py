"""Real-signal tests of blind input isolation and frozen-table interpolation."""

import numpy as np
import torch

from .common import OUTPUT, ROOT, write_json
from .data import prepare
from .experiment import load_models
from .inference import build_tables, extended_template, interpolate, predict


@torch.no_grad()
def audit():
    record = prepare("sub-01", "staircase", 30)
    models = load_models("sub-02")
    tables = build_tables(record, models)
    interpolation_errors = []
    for (morph, _), table in zip(models, tables, strict=True):
        for value in (0.023, 0.147, 0.489, 0.777, 0.999):
            q = torch.tensor(value)
            raw = morph(record["template"], q.expand(len(record["template"])))
            direct = extended_template(raw, record["projection"], record["amplitude"])
            approximate = interpolate(table, q)
            error = (approximate - direct).norm(dim=1) / direct.norm(dim=1).clamp_min(1e-6)
            interpolation_errors.append(float(error.max()))
    if max(interpolation_errors) > 0.01:
        raise ValueError(
            "Template lookup approximation exceeds 1% relative projected-template error"
        )
    batch = record["config"].batch_size
    boundary, limit = 100 * batch, 160 * batch
    record["x"] = record["x"][:limit].clone()
    record["features"] = record["features"][:160].clone()
    record["q"] = record["q"][:160].clone()
    original = predict(record, models, mode="b", tables=tables)
    changed = dict(record)
    # All actual kinematics change, even in the prefix; blind inference must ignore it.
    changed["q"] = 1 - record["q"]
    blind = predict(changed, models, mode="b", tables=tables)
    blind_isolated = bool(
        np.array_equal(original["spikes"], blind["spikes"])
        and np.array_equal(original["sources"], blind["sources"])
    )
    changed["x"] = record["x"].clone()
    changed["features"] = record["features"].clone()
    changed["x"][boundary:] *= -2
    changed["features"][100:] *= -2
    future = predict(changed, models, mode="b", tables=tables)
    prefix = bool(
        np.array_equal(original["spikes"][:boundary], future["spikes"][:boundary])
        and np.array_equal(original["sources"][:boundary], future["sources"][:boundary])
    )
    if not blind_isolated or not prefix:
        raise ValueError("Blind input isolation or future-prefix audit failed")
    result = dict(
        scope="sub-01 training record; causal batches after original offline preprocessing",
        blind_spikes_and_sources_ignore_all_true_angles=blind_isolated,
        future_perturbation_preserves_prefix=prefix,
        prefix_samples=boundary,
        maximum_interpolation_relative_error=max(interpolation_errors),
        original_post_calibration_true_spikes_loaded=False,
    )
    write_json(OUTPUT / "audit.json", result)
    write_json(ROOT / "docs/benchmarks/stage_b_audit.json", result)
    print(result)
