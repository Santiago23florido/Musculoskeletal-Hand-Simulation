# Multichannel template recovery with Pareto fallback

This experiment follows the 97-parameter residual. It tests three mechanisms:
additional physical correction directions, reidentification of inactive units,
and preservation of discharges on returning to the calibration posture.
A constant correction selected without RoA provides a control for the value
of learned conditioning.

This **analytic control inspired by MoCA-K** uses the measured angle. A full
MoCA implementation, a new Stage A run, and an EMG-only model are outside its
scope. It does not implement the guide's FiLM morphing CNN, supervision with
true MUAP trajectories, full FIFO covariance, peel-off, or units absent from
calibration.

## Rationale and decisions made before testing

The research guide, §§5.2–5.4 and algorithm 1, proposes explicit MUAP
representation and filter synthesis from the mixture. The signal model sums
discharge trains convolved with varying multichannel MUAPs. Adapting to this
mixture is also the problem studied by the
[Mendez Guerra et al. benchmark](https://doi.org/10.1088/1741-2552/ad5ebf).
[Klotz and Rohlén](https://arxiv.org/abs/2502.04065) develop inversion theory and
errors arising from MUAP similarity, noise, and conditioning.
[Nawab et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2932793/) provide
precedents for template creation, search, and updates. These sources motivate
mechanisms without proving this experiment's effectiveness.

### 1. Templates and additional directions

The 100 channels are spatially whitened using calibration covariance:

`S = (C_cal + 0.05 trace(C_cal)/100 I)^(-1/2)`.

The original ten lags then give 1,000 coordinates. This separable covariance
approximation differs from full adaptive whitening in Pareto, which remains an
independent branch. Templates `H0_j` average extended EMG at predicted calibration
discharges. They are multichannel estimates aligned to those peaks, without
providing the simulator's true MUAPs or necessarily their full temporal shapes.

With `q = (angle - calibration angle)/40`, online fitting uses

`H_j(q) = H0_j + q B_j`.

Fitting `B_j` solves ridge regression through sufficient statistics, with
pseudocount 20. It incorporates only previous Pareto peaks whose similarity to
the retained template exceeds a calibration threshold. The batch decision is
emitted **before** incorporating that batch's observations. `H0` is immutable
and `||H_j(q)-H0_j|| <= 0.5 ||H0_j||`. Linear dependence on angle is a limited
local hypothesis, without an established physiological law.

With normalized columns of `H`, filters are

`F(H) = H (H.T H + 0.1 I)^(-1)`.

This is the regularized inverse of the covariance model `H H.T + 0.1 I`,
computed through a small solve shared across units. Its matrix has eigenvalues
of at least 0.1, ensuring a unique solve. This does not establish identifiability
of every unit, global system invertibility, or improved RoA.

The physical direction added to the earlier residual is

`r = z [F(H(q)) - F(H0)] gain_cal`.

Its per-unit RMS is capped at 0.25 times the calibration-source RMS. This tests
multichannel directions beyond the line between fixed and Pareto filters.
The RMS bound limits batch output displacement; it does not guarantee lower
discharge error.

### 2. Explicit recovery

A separate detector receives `F(H(q))` sources, with amplitudes and centroids
estimated only during calibration. It retains a unit's template even if Pareto
produces no peaks for it. It may propose a discharge after the main branch has
gone at least 250 ms without detecting that unit, provided that:

- Multichannel evidence exceeds the tenth percentile of its calibration peaks.
- Similarity exceeds that of every other calibrated identity; ties are rejected.
- The original 10 ms minimum distance from emitted discharges is respected,
  across batches and relative to main-branch peaks in the current batch.

No event is added without evidence. Recovery does not rely on contrast-loss
gradients reaching an inactive unit. It is a search/detection decision, with
its false positives measured separately from differentiable training losses.

### 3. Reference-posture protection

The earlier neural correction and physical direction are multiplied by
`clip((|q|-0.025)/0.1, 0, 1)`. When `|q|<=0.025` and normalized velocity
`|v|<=0.1`, the original discharges from the **full Pareto branch** are returned.
Altered secondary-detector centroids do not affect that fallback. The condition
depends on current kinematics, without condition names, known recording duration,
or future labels.

Exact equality with Pareto is checked on fallback samples. This cannot guarantee
absence of regression at static postures away from calibration, during
transitions, or in final phases of other data.

## Comparison and generalization

The same four outer subjects, `sub-02..05`, provide 80 records and 591 calibrated
instances. All eight earlier neural models retain their weights, combining the
two seeds as before. No new network is trained. Template regression is causal,
regularized adaptation to the recording, based on predictions rather than true
post-calibration discharges.

Each fold compares constants `[0, -0.075, -0.15, 0.075, 0.15]` using the prior
study's three validation conditions at 25 dB in a subject distinct from the
outer subject. The same replay and unsupervised objective apply, with zero
eligible, at least 0.5% improvement, and at most 0.01 regression per condition.
RoA is excluded from constant selection. The original Pareto fitting subject,
`sub-01`, is never presented as an outer subject. The 5 s and 20 s ramps and
15 dB are excluded from selection. The previous network retains its original
training.

| Method | Role |
| --- | --- |
| `pareto` | Full original reference |
| `previous` | Earlier neural residual, with weights and predictions retained |
| `constant` | Constant correction selected without labels |
| `protected` | Earlier network with attenuation and reference fallback |
| `physical` | `protected` plus physical direction, without discharge rescue |
| `rescue` | Primary variant: `physical` plus explicit recovery |

The primary method is fixed before scoring and is not replaced by the best
ablation after examining RoA. Criteria require five-point gains on both 5/10 s
ramps, no loss in staircase, final phase, or final units ≥90%, a strictly higher
global mean, and no loss at 15 dB. Original evaluator alignment and 2 ms
tolerance are retained, including separate realignment within each phase.

**All original results had already been examined.** Subject separation prevents
direct fitting on the outer subject, but cannot remove prior design exposure
to the benchmark. Independent confirmation requires newly reserved records and
other generators or human data. The method uses no ELMo component or evidence
from its text experiments.

## Reproduction and controls

Use `.venv-benchmarks`, the existing data, and the preceding residual study.
Set `TEMPLATE_RESCUE_OUTPUT` to a fresh directory for another run.

```powershell
.\.venv-benchmarks\Scripts\python.exe -m pytest -q benchmarks/template_rescue/test_rescue.py
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.template_rescue.run audit
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.template_rescue.run register
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.template_rescue.run select
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.template_rescue.run predict
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.template_rescue.run score
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.template_rescue.run report
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.analyze_template_rescue
```

`predict --subject sub-02` generates one fold; after all four, run `seal` before
`score`. All 320 new prediction files are sealed before opening labels to score
the first. Earlier neural predictions are compared unchanged. Hashes cover
code, protocol, previous models, calibrations, signals, angles, and predictions.

Numerical tests cover the regularized inverse against a dense solve, degenerate
channels, morph anchoring and bounds, unit permutation, ambiguous-identity
rejection, evidence-based recovery, and cross-batch refractory behavior.
A real perturbation of future batches checks that the first 24,480 samples of
sources, evidence, and discharges remain identical. Causality applies to
100 ms batches **after the original offline preprocessing**; fully online raw
EMG preprocessing is not established.

Timing of the additional branch excludes Pareto adaptation, preprocessing, and
disk reads, so it cannot establish the guide's full MoCA latency. The
[summary](../../docs/benchmarks/template_rescue_summary.json) retains results
and failures.

## Measured results

The primary method did not meet the acceptance criteria. Mean RoA on the
80 outer-test records is 81.4166%, compared with Pareto's 80.8885% and the
previous residual's 81.6896%. Differences from Pareto are +2.8785 points on
5 s ramps, +0.2665 on 10 s ramps, −0.1845 on staircase, and +1.0239 at 15 dB.
**Final retention returns to 511 instances at or above 90%, matching Pareto and
improving on the earlier residual's 501.** Final-phase RoA loses 0.00025545
points, failing the strict no-loss criterion. Original thresholds remain in
force.

| Variant | Mean RoA (%) | Final instances ≥90% |
| --- | ---: | ---: |
| Pareto | 80.8885 | 511 |
| Previous residual | 81.6896 | 501 |
| Selected constant | 80.8121 | 510 |
| Protection | 81.6706 | **511** |
| Protection + physical direction | 81.4984 | 511 |
| Primary with recovery | 81.4166 | **511** |

**Protection restores the final-instance retention lost by the earlier model.**
The physical direction loses 0.1723 points relative to protection; added recovery
loses another 0.0818. Rescue adds 6,792 events, with net +2,082 TP and +4,710 FP
under each method's own alignment. These aggregate changes do not independently
label each added event. Mean precision falls from 91.0366% to 90.1264% relative
to the branch without rescue. Linear template estimation and its identity
checks were insufficient.

The final-boundary analysis finds five sample-times differing from Pareto,
in five instances from four records, within the phase's first 54 ms. The envelope
is already zero, but fallback also requires low velocity; that first batch
retains a previously altered detector. The threshold is not changed after
examining test results. Rounding does not satisfy the original criterion, and
the primary method is retained rather than replaced by an ablation.

![Differences from Pareto and final retention](../../docs/images/template-rescue-deltas.png)

All 51 benchmark tests pass. Results remain exploratory, without independent
evidence beyond these data. See the
[summary](../../docs/benchmarks/template_rescue_summary.json),
[component analysis](../../docs/benchmarks/template_rescue_analysis.json),
and [boundary diagnostic](../../docs/benchmarks/template_rescue_boundary.json).
