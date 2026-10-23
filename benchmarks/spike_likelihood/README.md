# Discharge detection with spike likelihood

After the [Stage B CNN failed](../stage_b/README.md), this variant learns the
decision being evaluated: whether to accept or reject each unit's discharge
candidates. It retains adaptive Pareto sources and fixed sources from the same
benchmark as observations. **The improvement is in event decoding; recovery of
true MUAP trajectories remains unverified.** The
[separation](../stage_b_separation/README.md) and
[geometry](../stage_b_geometry/README.md) pilots document the preceding attempts.

## Measured results

**Eight CUDA training runs completed, and 160 predictions were saved before
scoring**, with weights and thresholds frozen. Evaluation uses the same
80 outer-test records and 591 calibrated instances as the preceding study.

| Method | Global RoA | 5 s ramps | 10 s ramps | Recall | Final ≥90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pareto | 80.89% | 65.81% | 77.14% | 83.16% | 511 |
| **Likelihood, primary** | **83.51%** | **72.33%** | **80.05%** | **87.74%** | **520** |
| Rescue union | 83.27% | 72.08% | 79.82% | 87.86% | 506 |
| Previous morphological Stage B | 37.26% | 33.78% | 34.93% | 39.30% | 573 |

The gain over Pareto is **2.62 percentage points globally and 6.52 points on
ramp5s**, with 2.91 points on ramp10s and 2.70 at 15 dB. All four subjects and
all five conditions improve. There are 502 improving instances and 89 declining
instances. The descriptive subject-bootstrap interval for the global gain is
[+2.21, +2.98] points; there are only four subjects, and training is shared
between folds.

Mean precision changes from 91.47% to 91.10%. Recall improves at the cost of
false positives. Evaluator counts sum to +30,653 TP and +10,285 FP, using each
method's own lag; they do not independently label each added event. Learned
event removal contributes 0.24 points compared with retaining all earlier
events through a union.

At the end of the recording, 16 instances cross above 90% and seven fall below
it, for a net gain of nine. Some units retained by Pareto are therefore lost.
Mean final RoA rises from 87.99% to 95.25%. Global median RoA falls slightly,
from 95.29% to 95.07%. This reports the cost to strong units alongside the gain
for weaker ones.

The full guide targets remain unmet: global RoA is 83.51% against 85%, ramp5s
is 72.33% against 80%, and 59.73% of instances achieve global RoA ≥90%, against
a target of 70%. The archived JSON retains its original status. **Event detection
improves on this benchmark**, while the earlier morphology CNN remains
unsuccessful.

Results: [summary](../../docs/benchmarks/spike_likelihood_summary.json),
[per-instance CSV](../../docs/benchmarks/spike_likelihood_per_unit.csv),
[descriptive analysis](../../docs/benchmarks/spike_likelihood_analysis.json),
[audit](../../docs/benchmarks/spike_likelihood_audit.json), and
[frozen selection](../../docs/benchmarks/spike_likelihood_locked.json).

![Corrected decoder comparison on the same outer-test records](../../docs/images/spike-likelihood.png)

## Rationale

Squared error on a waveform or continuous source does not determine the right
detection threshold. It may decrease without recovering events. Here, the label
is `y=1` when a candidate matches a true discharge within the 2 ms tolerance,
using only the lag measured during initial calibration.

Training uses binary log loss:

\[
L=-\mathbb E[y\log p_\theta+(1-y)\log(1-p_\theta)].
\]

Conditional expected loss is minimized at `p=P(y=1|z)`. This is a strictly proper
scoring rule under its training distribution; see
[Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437).
That supports learning discharge probabilities, without guaranteeing
calibration outside the training distribution or high RoA. The threshold is
selected on validation event metrics, following the distinction between
probability estimation and F1/Jaccard decision selection in
[Lipton et al., 2014](https://arxiv.org/abs/1402.1892).

The model corrects Pareto's log odds:

\[
\operatorname{logit}p=\ell_0+\eta\,4\tanh(f_\theta(z)),
\qquad \ell_0=\begin{cases}\log9&\text{Pareto event}\\-\log9&\text{other candidate.}\end{cases}
\]

`f` is a 19→8→1 MLP with **169 parameters shared across all units and subjects**.
The final layer starts at zero. Zero correction and `η=0` exactly reproduce
Pareto, including its initial prediction. Bounding the residual regularizes
how far the initial probabilities can change. The starting values are a
reference point; 0.9/0.1 are not claimed to be empirically calibrated probabilities.

The nineteen inputs describe calibration-normalized amplitude, local temporal
shape, agreement between the two views, source competition, calibration
SIL/rate, and six batch EMG statistics. **Inference uses no angles, true
velocity, absolute clock, or trainable identifiers.** The two kinematic columns
of the older cache are explicitly discarded.

Candidates include all Pareto events and weak maxima from both sources. The
shape window is limited to the current batch of about 100 ms. Final decisions
apply within-batch non-maximum suppression at the original detector's minimum
distance. Original preprocessing remains offline; causality applies to blocks
after that step.

## Fitting and generalization

The subject-based outer validation is retained: four outer subjects,
`sub-02` through `sub-05`, one validation subject per fold, and three training
subjects, always including `sub-01`. Training uses
staircase/ramp40s/ramp10s at 30/20 dB, with selection at 25 dB. Ramp5s, ramp20s,
and 15 dB are excluded from fitting and weight/threshold selection for each
fold.

Training labels come from samples after the first five seconds of the training
subjects. Classes are not reweighted and negatives are not discarded, preserving
their observed frequency. Units are balanced so a source with many noise peaks
does not dominate the objective. Each unit's lag comes from its five-second
calibration, rather than alignment of the full training record after prediction.

The budget is ten epochs and two seeds, using AdamW with learning rate 0.001,
weight decay 0.01, input noise 0.02, and gradient clipping at one. Validation
runs every two epochs. The initial identity candidate is eligible. Selection
uses mean per-unit validation Jaccard with calibration lags and a fixed set of
seven thresholds. Logits from both seeds are then averaged, and `η∈{0,0.5,1}`
and the threshold are chosen using validation only.

All eight models and ensemble decisions are frozen before outer-test prediction.
The **160 predictions** comprise posterior decisions and an ablation that adds
their events while retaining original ones. All are sealed before scoring
begins. Pareto and the earlier Stage B are references on the same **591 instances
from 80 records**. Outer scoring uses the full original evaluator, including
its original lag search.

This benchmark had already been explored and motivated the correction. New
decisions use training/validation, but the result remains exploratory transfer
within the same generator. A small parameter count and absence of identifiers
do not establish independent generalization.

## Reproduction

Use the existing CUDA and CPU environments. Earlier data, sources, and caches
remain frozen. Output defaults to `outputs/spike_likelihood`;
`SPIKE_LIKELIHOOD_OUTPUT` selects a different directory.

```powershell
foreach ($subject in 'sub-01','sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run prepare --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  foreach ($trialSeed in 0,1) {
    .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.spike_likelihood.run train --subject $subject --seed $trialSeed
  }
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run freeze
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run audit
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run predict --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run start-scoring
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run score --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run report
```

Subjects can be prepared, predicted, or scored in independent processes;
scoring is distributed only after validating the complete manifest. The report
requires eighty records and 591 unique keys per method. Compact results are
copied with the prefix `docs/benchmarks/spike_likelihood_`.

Recorded timing starts after candidate computation and measures CPU likelihood,
decisions, and output persistence. It excludes feature extraction and prior
decomposition. The full 22 ms budget from raw EMG, true 1/2.5 s ramps, and an
independent generator remain to be evaluated. Single-source SIL is not assigned
to decisions that combine two views.
