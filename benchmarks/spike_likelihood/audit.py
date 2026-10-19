import json

import numpy as np

from .common import OUTPUT, ROOT, write_json
from .data import decode, prepare
from .experiment import load_models, logits


def audit():
    record = prepare("sub-01", "staircase", 30)
    models, selection = load_models("sub-02")

    def predict(r):
        z = np.mean([logits(m, r, "cpu") for m in models], axis=0)
        prior = (r["primary"].astype(np.float32) * 2 - 1) * 2.1972245773362196
        p = 1 / (1 + np.exp(-(prior + selection["strength"] * (z - prior))))
        return decode(r, p, selection["threshold"])

    original = predict(record)
    changed = dict(record)
    changed["features"] = record["features"].copy()
    boundary = 100 * record["batch"]
    changed["features"][record["time"] >= boundary] *= -3
    altered = predict(changed)
    prefix = bool(np.array_equal(original[:boundary], altered[:boundary]))
    metadata = [json.loads(p.read_text()) for p in (OUTPUT / "features").glob("*/metadata.json")]
    identity = all(m["baseline_spike_disagreements"] == 0 for m in metadata)
    if not prefix or not identity:
        raise ValueError("Future-prefix or identity audit failed")
    result = dict(
        scope="training record; candidate EMG features after original offline source preprocessing",
        future_feature_perturbation_preserves_prefix=prefix,
        prefix_samples=boundary,
        identity_spike_disagreements=0,
        identity_records=len(metadata),
        input_features=19,
        true_angle_inputs=False,
        postcalibration_true_spikes_loaded=False,
    )
    write_json(OUTPUT / "audit.json", result)
    write_json(ROOT / "docs/benchmarks/spike_likelihood_audit.json", result)
    print(result)
