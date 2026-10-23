# Analytic decoding with self-cancellation correction

**This proposal was withdrawn.** Switching to the FIFO regime with unit-response
normalization causes amplitude loss relative to the detector threshold; the
collapse reproduces on sub-01. Excluding only the current block recovers some
RoA, but the remaining deficit has not been fully explained. The
[correction and CPU audit](../../docs/benchmarks/robust_mvdr_correction.md)
records measurements, sensitivity, and limits of this attribution.

The [original report](../../docs/benchmarks/robust_mvdr_diagnostic.md) retains
results for 80 records and 591 instances: static_subtract achieved 5.22%.
The four decision rules cannot test hypothesis F1 because their control shares
the defect. Registered executable code, predictions, and scored results remain
unchanged; sub-02 through sub-05 are not reevaluated.

The remaining sections document the historical proposal from the
[independent review](../../docs/benchmarks/independent_review.md), withdrawn
after the audit. It should not be rerun on the outer subjects.

## Original hypothesis, not confirmed

The proposed unit-response filter, `w = R⁻¹h / (hᵀR⁻¹h)`, is an MVDR/Capon
beamformer. Stage B divides by `sqrt(hᵀR⁻¹h)`; the amplitude difference matters.
`R` is observed-data FIFO covariance and **contains the target unit**.
When `h` mismatches the true MUAP, MVDR suppresses part of the unit itself:
the self-cancellation described by
[Cox, 1973](https://doi.org/10.1121/1.1913659) and addressed by robust Capon in
[Li, Stoica, and Wang, 2003](https://doi.org/10.1109/TSP.2003.812831).
Response to the true unit falls below the threshold calibrated at rest, while
other units remain suppressed. This predicts high precision and low recall,
as measured in Stage B:

| Stage B method | RoA | Precision | Recall | Final plateau |
| --- | ---: | ---: | ---: | ---: |
| Pareto | 80.89% | 91.47% | 83.16% | 87.99% |
| Analytic static template | 45.56% | 91.39% | 47.85% | 97.50% |
| MoCA-B | 37.26% | 92.17% | 39.30% | 97.32% |

The final plateau, where the template becomes correct again, exceeds Pareto.
The original hypothesis attributed the deficit away from that posture to
template mismatch; the later audit above limits that interpretation.

## Synthetic evidence used to set parameters

An independent *line-source* generator was used in place of BioMime
([`synthetic/generator.py`](synthetic/generator.py)), with NeuroMotion's length
laws, a 10×10 grid, and 20 dB noise. Calibration occurs at `s=1`, followed by a
10 s hold at shortened length `s₁`, without movement, to exclude covariance lag.
The twelve most separable units among roughly forty active units are decoded,
using templates interpolated between the posture's true MUAP (λ=0) and the
static calibration template (λ=1). RoA uses 2 ms tolerance. Entries show a fixed
MVDR threshold of 0.5 / unlabeled two-means threshold / best threshold.

| s₁, λ | Full `R` | Loading 0.1 | ρ=0.5 + loading 0.03 | ρ=1 + loading 0.1 |
| --- | --- | --- | --- | --- |
| 0.94, λ=0 | 0.999 / 0.999 / 0.986 | 0.912 / 0.990 / 0.984 | 0.963 / 0.997 / 0.990 | 0.919 / 0.991 / 0.981 |
| 0.94, λ=0.5 | 0.864 / 0.906 / 0.943 | 0.833 / 0.923 / 0.947 | 0.869 / 0.968 / 0.971 | 0.843 / 0.918 / 0.949 |
| 0.94, λ=1 | 0.585 / 0.897 / 0.935 | 0.853 / 0.972 / 0.972 | 0.848 / 0.982 / 0.976 | 0.931 / 0.966 / 0.964 |
| 0.91, λ=0.5 | 0.674 / 0.722 / 0.858 | 0.728 / 0.744 / 0.861 | 0.773 / 0.790 / 0.876 | 0.759 / 0.758 / 0.864 |
| 0.91, λ=1 | 0.312 / 0.770 / 0.854 | 0.730 / 0.851 / 0.897 | 0.650 / 0.920 / 0.940 | 0.848 / 0.888 / 0.918 |

With covariance **excluding the target unit** (an oracle bound unavailable in
practice), the static template reaches 0.997 at `s₁=0.94` and 0.870 at `s₁=0.91`
with the fixed threshold. Separability is largely retained (best-threshold
RoA ≥0.85 in all cells); response amplitude fails relative to a fixed threshold.
The Stage B covariance-model mixture with α=0.5 yields fixed-threshold RoA of
0.74–0.76 even with the exact template in this generator. Recall stays at 0.99,
so false positives drive the loss: the twelve-unit model no longer suppresses
uncalibrated units. Blanking the unit's own detections provides little benefit
(0.677 versus 0.585 at `s₁=0.94`, λ=1).

ρ=0.5 with loading 0.03 was chosen as the primary method: across nine cells
(`s₁` = 0.96, 0.94, and 0.91), it is best or within 0.3 percentage points of
the best under two-means thresholding, the closest analogue to adaptive
detector centroids. With an exact template, the cost is 1.9–5.4 points under
the fixed threshold and at most 0.3 under two-means. ρ=1 with loading 0.1 wins
with a fixed threshold under large mismatch and is retained as a preregistered
ablation. Parameters were set using synthetic data only, before opening any
FDSI recording in this review.

## Filter

For each unit `j`, its full template across 73 lags is `C_j`. The zero-lag row
is exactly Stage B's `extended_template`:

```text
A   = R + loading · mean(eig(R)) · I
R_j = A − ρ · rate_j · C_jᵀ C_j
w_j = R_j⁻¹ h_j / (h_jᵀ R_j⁻¹ h_j)
```

`R_j⁻¹` is obtained with Woodbury from a single Cholesky factorization of `A`.
Eigenvalues of the inner matrix `I − UᵀA⁻¹U` are clipped at 0.05, preventing
subtraction from making covariance indefinite even with an incorrect template.
Rate is initialized from calibration and updated by EMA of detector-accepted
discharges, as in Stage B.

## Preregistered methods

| Method | State | ρ | Loading | Purpose |
| --- | --- | ---: | ---: | --- |
| `static_subtract` (primary) | Calibration template | 0.5 | 0.03 | Test whether correcting self-cancellation is sufficient |
| `static_subtract_strong` | Calibration template | 1.0 | 0.1 | More aggressive ablation |
| `static_loaded` | Calibration template | 0 | 0.1 | Robust Capon only |
| `static_fifo_mvdr` | Calibration template | 0 | 0.001 | Control: FIFO MVDR without α mixing |
| `moca_k_subtract` | Frozen morph, measured angle | 0.5 | 0.03 | Morph value after self-cancellation correction |
| `moca_b_subtract` | Frozen morph and observer | 0.5 | 0.03 | Same, with EMG-only state |

Comparators are Pareto and, when present and unchanged, sealed Stage B predictions
(`stage_b_static_analytic`, `stage_b_moca_k`, `stage_b_moca_b`), on the same
591 instances.

The interpretation was declared before prediction:

1. If `static_subtract` exceeds `stage_b_static_analytic` in recall and RoA,
   especially during angular `moving` phases and ramp5s, self-cancellation was
   a real FDSI bottleneck.
2. If `static_subtract ≈ static_fifo_mvdr`, self-cancellation does not explain
   the FDSI deficit and the diagnosis needs revision.
3. `moca_*_subtract − static_subtract` measures the frozen morph's value after
   self-cancellation removal. Values ≤0 are interpreted alongside
   `target-reliability`.
4. This filter alone was not expected to beat Pareto. A later step would add
   [spike likelihood](../spike_likelihood/README.md) or an unlabeled adaptive
   threshold.

Subjects `sub-02`–`sub-05` had already been scored in five preceding experiments.
Even without fitting on them here, the result is **diagnostic and exploratory**.
Any improvement requires confirmation with new records from another generator
or new NeuroMotion simulations.

## Audits

- `phase-audit` reads angles only. It reports each record's angle at 5 and 10 s
  and the moving fraction within 0–10 s. Triangular `first_hold` spans 0–10 s,
  matching reference `get_triangular_phases`; a nonzero moving fraction means
  this label contains movement. The evaluator also adds angle-defined phases:
  reference rest after calibration, movement (>0.5°/s), and plateaus away from
  the reference posture.
- `target-reliability` uses only `sub-01`, a training subject in every Stage B
  fold. It splits Γ's posture-specific STA targets into even/odd halves and
  compares their noise with their change from the calibration template, both
  in raw coordinates and separation geometry. A change/noise ratio near one
  means those targets lack a learnable posture signal.

## CPU reproduction

The historical commands below require `.venv-benchmarks`, FDSI data, and Stage B
outputs in `outputs/stage_b_main`: cached inputs, `protocol.json`, `locked.json`,
and morph/observer weights under each fold and seed. Without those weights,
set `ROBUST_MVDR_STATIC_ONLY=1` for **every** command; that choice is part of the
registered specification. The withdrawal above still applies to outer-subject
runs.

### Compatibility with frozen weights

Publication made the PDF hash optional in `Stage B register` after weights
were frozen. The original loader rejects that provenance change. `frozen.py`
preserves the protocol, archived sources, and earlier weights: it verifies
their hashes, all sixteen checkpoints, and current code. The only permitted
difference is the expression making `guide_sha256` optional within `register`;
the rest of the syntax tree must be identical. Numerical, checkpoint, or
protocol changes are rejected. Original classes are instantiated on CPU with
their exact parameters and no training. New registration records this provenance,
including the metadata exception.

```powershell
.venv-benchmarks/Scripts/python.exe -m pytest benchmarks/robust_mvdr -q
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run phase-audit
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run target-reliability
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run predict --subject $subject --threads 4
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run score
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run report
```

Output defaults to `outputs/robust_mvdr`; `ROBUST_MVDR_OUTPUT` selects another
directory. `register` records hashes of its own code and Stage B dependencies
and refuses to overwrite a protocol. `predict` does not open true outer-subject
discharges. `seal` requires all 80 × 6 predictions; `score` checks their hashes
before scoring and verifies that the evaluator reproduces Pareto's saved RoA.
Subjects can be predicted in parallel processes.

Subtraction synthesis takes about 9 ms per 100 ms batch with eight units and
D=512 (four threads, measured on Linux), in addition to Stage B's 8–11 ms loop.
Stage B took a median of 51 s per record with four methods (1.1 h total).
The historical estimate here was 1.5–3 min per record with six methods, or
roughly 2–4 h for all 80 records.

`report` writes `report/summary.json` and `report/per_unit.csv`, with means by
condition, SNR, and phase (fixed and angular), plus differences from Pareto and
the Stage B static template. Compact results use the prefix
`docs/benchmarks/robust_mvdr_`.

The synthetic study can be reproduced without external data:

```powershell
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.synthetic.study all --output docs/benchmarks
```

It writes `docs/benchmarks/robust_mvdr_synthetic_*.json` containing the table
values above. Seeds are fixed; a different BLAS thread count can change an
isolated decision through rounding and move a cell by ±0.001.
