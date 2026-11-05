"""Post-hoc, read-only: P2 contrasts that the protocol did not register.

Same rows, clusters, bootstrap (2000 draws, seed 2026) and metrics as context_c4.score;
writes only to the scratchpad. Labelled post-hoc in the journal.
"""

import json
import sys
from collections import defaultdict

import numpy as np

from benchmarks.pareto_mc.common import VALIDATION, records, stub
from benchmarks.pareto_mc.context_c4 import FOLDER, SEEDS
from benchmarks.pareto_mc.events import descriptor
from benchmarks.pareto_mc.l2_ablation import combine, metrics

ARMS = ("wide24", "set24", "setrec24", "mdiff24")
KEEP = ("global_roa_pct", "ramp5s_roa_pct", "precision_pct", "recall_pct", "hold_away", "moving")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(name, sign):
    return {f"{name}_seed{s}": sign / 3 for s in SEEDS}


table = {
    "set24_vs_wide24": {**mean("set24", 1), **mean("wide24", -1)},
    "setrec24_vs_wide24": {**mean("setrec24", 1), **mean("wide24", -1)},
    "mdiff24_vs_wide24": {**mean("mdiff24", 1), **mean("wide24", -1)},
}
rows, clusters = defaultdict(dict), defaultdict(list)
for case in records(VALIDATION):
    calibration = descriptor(*case, phase="validation")["calibration"]
    for unit, index in enumerate(calibration.gt_matched_indices):
        clusters[(case[1], int(index))].append((case[1], case[2], unit))
for arm in ARMS:
    for s in SEEDS:
        for case in records(VALIDATION):
            for row in read(FOLDER / f"scores/{arm}/seed{s}/{stub(*case)}.json")["rows"]:
                rows[f"{arm}_seed{s}"][(row["condition"], row["snr_db"], row["unit"])] = row
keys, names = sorted(rows["wide24_seed0"]), sorted(clusters)


def summary(selected):
    values = {label: metrics([by_key[k] for k in selected]) for label, by_key in rows.items()}
    return {name: combine(w, values) for name, w in table.items()}


point = summary(keys)
rng = np.random.default_rng(2026)
draws = defaultdict(lambda: defaultdict(list))
for _ in range(2000):
    pick = rng.integers(0, len(names), len(names))
    for name, effect in summary([k for i in pick for k in clusters[names[i]]]).items():
        for metric, value in effect.items():
            draws[name][metric].append(np.nan if value is None else value)
out = {
    name: {
        m: dict(
            estimate=round(v, 3),
            ci95=[round(float(x), 3) for x in np.nanpercentile(draws[name][m], [2.5, 97.5])],
        )
        for m, v in effect.items()
        if m in KEEP
    }
    for name, effect in point.items()
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
