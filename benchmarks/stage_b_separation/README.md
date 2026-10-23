# Stage B: separation geometry correction

**This pilot stopped during validation, before outer-test inference.** The
refined CNN reduces projected error, but its source risk is worse than the
identity in completed folds. Its weights were discarded. The subsequent
experiment uses [discharge labels](../spike_likelihood/README.md).

The pilot addresses the mismatch observed in the
[preceding variant](../stage_b/README.md), where lower full-waveform NMSE
worsened projection into the separator. It retains the same records, templates,
projection, Γ architecture, and frozen observers, changing Γ's normalization
and objective. Previous experiments and results remain intact.

## Derivation

Let `x=h_i s_i+interference`, with positive covariance R. The earlier direction,
`R⁻¹h/sqrt(hᵀR⁻¹h)`, has self-response `sqrt(hᵀR⁻¹h)`, which varies with posture
and covariance. Scale can therefore change even with a correct template, while
the detector starts with centroids calibrated at one posture.

**MVDR** imposes unit self-response:

\[
\min_w w^\top Rw\quad\text{subject to }w^\top h=1,
\qquad w=\frac{R^{-1}h}{h^\top R^{-1}h}.
\]

Lagrange multipliers give the solution. It preserves the separation direction
and fixes the expected amplitude of the modeled event. Additional gain is
calibrated once using the first five seconds. The original SD detector is
retained. Sources: [Capon, 1969](https://doi.org/10.1109/PROC.1969.7278) and
[MVDR documentation](https://www.mathworks.com/help/phased/ref/mvdrweights.html).
MVDR depends on template accuracy: its constraint applies to the estimated
model and cannot guarantee unit response for an incorrectly estimated unit.

For training, `h*` is the trajectory estimated from clean EMG and true discharges
of permitted subjects. C is observed projected-EMG covariance in a posture
neighborhood, and `R* = L*L*ᵀ` is its version with the same physical shrinkage.
The extended waveform is penalized **in the separation geometry**:

\[
L_p=\frac{\|L_*^{-1}(\widehat h-h_*)\|^2}{\|L_*^{-1}h_*\|^2}.
\]

This positive term controls direction and amplitude. Compared with NMSE over
6,400 waveform samples, it weights error after the projection and covariance
used by the separator. A differentiable source risk is added:

\[
L_s=\frac{(\widehat w-w_*)^\top C(\widehat w-w_*)}{w_*^\top Cw_*}
=\frac{\mathbb E[(x^\top\widehat w-x^\top w_*)^2]}
{\mathbb E[(x^\top w_*)^2]}.
\]

The equality holds for centered x with covariance C. It penalizes actual source
error relative to the teacher separator, including interference contributions
represented in C. Gradients pass through the covariance model, Cholesky, and
MVDR normalization. Source risk is distinct from RoA: the teacher uses estimated
templates and can be wrong.

The fixed objective is `Lp + 0.25 Ls + 0.1 Lang + 0.01 Lwave + 0.001 Lsize`.
Lang compares whitened directions; Lwave retains weak full-waveform
regularization. Retaining Lp prevents a local output-suppressing solution from
reducing Ls without learning identity. Exact identity at q=0 and the correction
norm bound of two remain.

## Protocol and controls

The pilot reuses four outer folds, two seeds, and 42 permitted supervision
records: staircase/ramp40s/ramp10s, SNR 30/20 for fitting and 25 for validation.
Each record has nine states with targets for all its units. Each model excludes
its outer subject from fitting; ramp5s/ramp20s/15 dB are excluded from fitting
and selection. Covariances pool training/validation samples by posture.
Inference retains the causal ten-batch FIFO after offline preprocessing.

Γ is refined for twenty epochs on GPU using AdamW, learning rate 0.0003,
weight decay 0.001, gradient clipping at one, and 3% input perturbation.
Candidates include identity without correction, previous unrefined weights,
and refinement epochs. Selection uses **unit-weighted validation source risk**,
without outer-test RoA. The previous B observer remains frozen, with the same
state per seed. Parameter counts stay at 3,137 for Γ and 10,537 for the observer.

The planned four new methods are corrected Γ with B state, the same Γ with K
state, old Γ with MVDR to isolate normalization, and a static template with MVDR
to isolate learning. Pareto and the earlier B are references on the same
591 instances. The protocol requires all 320 new predictions to be saved before
the first score, with weight selection and source snapshots frozen by hashes.
The pilot stopped before this evaluation.

Earlier results motivated the correction and had already exposed this benchmark.
New selection decisions use training/validation, but any measured transfer
would remain exploratory within the same generator. Independent confirmation
requires new data. Resampling does not supply true 1/2.5 s ramps, and complete
latency from raw EMG is not established.

## Execution

Use the preceding experiment's CUDA environment, `.venv-stage-b-gpu`, and CPU
evaluation environment, `.venv-benchmarks`. No dependency reinstallation is
needed. Output defaults to `outputs/stage_b_separation`;
`STAGE_B_SEPARATION_OUTPUT` selects a fresh directory.

```powershell
foreach ($subject in 'sub-01','sub-02','sub-03','sub-04','sub-05') {
  .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.stage_b_separation.run prepare --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  foreach ($trialSeed in 0,1) {
    .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.stage_b_separation.run train --subject $subject --seed $trialSeed
  }
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run freeze
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run audit
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run predict --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run start-scoring
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run score --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run report
```

After scoring starts, the four subjects can be distributed across independent
processes. The report requires eighty records and 591 unique instances per
method. Compact results would be exported with the prefix
`docs/benchmarks/stage_b_separation_`; large arrays and weights are ignored.

The [geometry pilot](../stage_b_geometry/README.md) documents the subsequent
alternative. Both pilots stopped during validation, without outer evaluation.
