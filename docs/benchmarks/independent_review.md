# Independent review of the movement-robust decomposition work — 19 October 2026

Follow-up correction: the robust decoder is **withdrawn**. The
[CPU FIFO audit and correction](robust_mvdr_correction.md) reproduce its collapse
on sub-01, without revisiting the outer subjects. Original results and registered
executable sources remain intact. F1 is an unconfirmed hypothesis; F2's claim of
noise-dominated targets is refuted. Synthetic RoA magnitude depends on the record.

This review covers `main` at `18e0fc4` (74 commits after the MoCA plan): the Farina
reproduction, Stage A, the Pareto residual, template rescue, Stage B (MoCA-K/B) with
its separation and geometry pilots, and the spike-likelihood repair. It checks
whether each stage implements what it claims and whether its conclusion follows
from its evidence. It also corrects errors in the plan that motivated the work.

## Method

- Read the code of every stage and the reference `adapt_decomp` at the pinned commit.
- Recomputed the published summaries from the per-instance CSV files.
- Ran the suites on Linux with the pinned versions (torch 2.7.0, numpy 1.26.4).
  Benchmarks: 75 passed, 2 CUDA skipped. Physics: 46 passed, 1 skipped. These match
  the [publication record](../publication_validation.md).
- Ran controlled experiments on an independent line-source generator, not BioMime.
  The code is in [`benchmarks/robust_mvdr/synthetic`](../../benchmarks/robust_mvdr/synthetic)
  and the outputs in `robust_mvdr_synthetic_*.json`.
- FDSI data were unavailable during the initial review. The later registered
  experiment completed; the FIFO follow-up used only existing sub-01 caches at
  30 dB, with no training or further outer-subject evaluation.

## Verdict by stage

| Stage | Implementation | Conclusion | Main issue |
| --- | --- | --- | --- |
| Farina reproduction | Correct | Valid | None |
| Stage A | Correct | Valid no-go | Design: rank-3 correction cannot represent the drift (F3) |
| Pareto residual | Correct | Valid, but mislabelled | Uses measured angle and velocity, so it is K-type (F4) |
| Template rescue | Correct | Valid no-go | `F(H)` ignores uncalibrated interference (F6) |
| Stage B MoCA | Correct | No-go valid; cause unconfirmed | Self-cancellation is a hypothesis (F1); target-noise claim refuted (F2) |
| Separation/geometry pilots | Correct | Valid stop | Metric changed; remaining failure cause unconfirmed |
| Spike likelihood | Correct, no leakage | Valid | Gain is mostly added candidates (F7) |
| Robust MVDR proposal | Defective FIFO/detector design | Withdrawn | Shared control collapse cannot test F1; reproduced on sub-01 |
| Phase reporting, all stages | Reproduces the reference | Misleading for ramps | "First hold" contains movement (F5) |

## Findings

### F1 — Stage B self-cancellation: an unconfirmed hypothesis

The original high-precision/low-recall observations do not identify their cause.
Stage B uses `R^-1 h / sqrt(h^T R^-1 h)`, while the proposed robust decoder uses
`R^-1 h / (h^T R^-1 h)`. Equal direction does not imply equal source amplitude.

The realistic replay instead identifies a defect in the proposed decoder: its
current block enters the ten-batch FIFO before it is filtered, and unit-response
normalization reduces true-spike amplitude against the calibrated detector.
Squared peaks must exceed adaptive centroid midpoints; the spike centroid can
update only when trusted spikes are detected. Low amplitude can prevent recovery.
A universal cutoff in amplitude ratio is not established by the per-unit tables.

Across three existing sub-01 30 dB caches, real FIFO post-5 s RoA is 0.1018%,
versus 42.8317% for real Stage B static. First-second amplitude ratios are 0.5157
and 0.7863. The six proposed outer-subject variants largely retain only copied
calibration discharges. Their control shares the defect, so their four decision
rules cannot discriminate F1. The remaining FDSI gap after excluding the current
block is documented in the correction; no exhaustive cause is established.

The criticism of alpha=0.5 mixing is withdrawn: in all eight realistic synthetic
cases, Stage B with output-power normalization and alpha=0.5 outperforms the
alpha=0 output-power pipeline. The original criticism used a different covariance
and threshold regime, and is not supported by this realistic pipeline.

**Lesson:** the original synthetic study used 10 s covariances and oracle or
two-means thresholds, not the short FIFO with the actual detector. Its end-to-end
fixture had D=48, not D=512. Synthetic magnitude is sensitive to the record; only
the declared qualitative amplitude claims are evaluated. No code was tuned to an
expected score. The proposal is withdrawn and F1 remains a hypothesis.

`frozen.py` is correct and unrelated: static controls do not load morph or observer
weights. No registered source or scored result was edited.

### F2 — Noise-dominated Gamma targets: refuted by the audit

The registered sub-01 split-half audit refutes the initial inference of
noise-dominated posture targets in separation geometry: the median moved-bin
signal-to-noise ratio is **30.9224**, projected target-noise NMSE is **0.034695**,
and projected change NMSE is **0.989930**. Substantial posture-dependent change
is reliable in the audited setting.

Raw-shape NMSE and separator-weighted errors are different metrics, but a large
projected change does not imply noisy targets. Reliability does not establish
Gamma's capacity or generalization. Do not retrain Gamma on a noise-dominated
explanation. Prioritize template tracking and event likelihood; future confirmation
needs fresh data.

**Evidence:** [unchanged target-reliability audit](robust_mvdr_target_reliability.json)
and [correction](robust_mvdr_correction.md).

### F3 — Stage A's rank-3 correction cannot fit whitening drift (major, plan error)

The plan confused a low-dimensional *state* with a low-rank *matrix correction*.
In exact generator covariances, calibration-ZCA coordinates drift with KL 16–133
nats between s = 0.99 and 0.88. After the best correction, the share of that KL
that remains is:

| Correction | KL left |
| --- | ---: |
| Rank 3 | 75–81% |
| Rank 18 | 43–53% |
| Rank 50 | 21–31% |
| Diagonal | 78–94% |
| Diagonal + rank 3 | 67–78% |

Measured consequences:

- Stage A matches the identity ablation: 57.83% against 57.23%.
- It collapses in the final hold: 43.5% against Pareto's 86.2%.
- It was trained on two sub-01 ramps only.

The no-go is correct.

**Change:** use full-rank, model-based covariance corrections instead of
low-rank whitening, or keep the KL adaptation.

### F4 — The Pareto residual uses measured kinematics (moderate, disclosure)

Two of its ten features are `q = angle/40` and its normalised velocity, read from
`*_angle.npz`. It is therefore a K-type method, not directly comparable with Pareto,
which never sees the angle. Its result is 81.69% against 80.89%. Template rescue
inherits these weights. The spike likelihood correctly drops these two columns
(`context[..., 2:]`); no leakage was found there.

**Change:** declared in the root README and in the residual README.

### F5 — "First hold" contains movement on triangular ramps (moderate, reporting)

The repository reproduces the reference `get_triangular_phases`. In it,
`first_iso = cal_end + iso_dur = 10 s` for triangular conditions and 5 s for the
staircase. Pareto's "first hold" RoA nevertheless falls with ramp speed: 98.3% at
40 s ramps and 73.5% at 5 s ramps. The static analytic decoder falls from 87.5% to
55.6%, and is 98.6% on the staircase. Ramp speed should not matter during a hold, so
the angle audit confirms movement in the 0–10 s window. Earlier phase comparisons ("MoCA loses
15 points in the first hold") are therefore early-movement comparisons.

**Confirmed:** ramp5s reaches 40 degrees at 10 s; its mean moving fraction in
0–10 s is 0.504053. `phase-audit` checks this from the angle traces alone. The robust
evaluator also reports phases defined by the angle (rest at reference after
calibration, moving above 0.5°/s, holds away from reference), next to the fixed
windows.

### F6 — Template rescue cannot help with active uncalibrated units (minor)

`F(H) = H(HᵀH + 0.1I)⁻¹` decorrelates calibrated templates only. It ignores the
covariance of active uncalibrated units, which the MVDR form handles. This is
consistent with its null effect (81.42%).

### F7 — Spike likelihood is valid; its gain is mostly recall (informative)

All figures recompute exactly from the CSV:

- RoA 83.51% against 80.89%; +6.52 points on 5 s ramps.
- All four subjects improve; 502 instances improve and 89 worsen.

The union variant (+2.38 points) shows that most of the gain comes from adding
weak candidates. Precision falls slightly, from 91.47% to 91.10%. This supports recovering weak candidates; it does not establish
self-cancellation as the cause of Stage B's deficit (F1).

### F8 — Benchmark CI coverage: resolved in PR #1

Ruff includes benchmarks, and the workflow tests the locked CPU environment on
Windows and Ubuntu with pinned adapt_decomp. FIFO replay equivalence, the D=512
amplitude regression, subject restrictions, and the missing-cache guard are in
the suite. Local validation: 105 passed, 2 CUDA tests skipped. See PR #1 checks
for remote status.

### F9 — The outer subjects are no longer a fresh test set (moderate, protocol)

`sub-02`–`sub-05` have now been reused by six experiments. Each one was registered
before scoring, but design choices made after seeing earlier results accumulate
selection risk. A new FDSI result is diagnostic, not confirmatory.

**Change:** confirm any improvement on new records, either new NeuroMotion
simulations or a different generator such as MUniverse.

## Corrections to the original plan

- **"Noise is not the bottleneck"** is withdrawn. The SNR means were taken over very
  different instance sets (71 instances at 15 dB against 332 at 30 dB). The flat curve
  may reflect which units survive calibration rather than robustness to noise.
- **"Rank 3 is enough"** is wrong for the whitening drift (F3). Rank 3 described the
  MUAP *state*, not the covariance change.
- **"The analytic filter is worse at the reference posture"** was a phase-label
  artefact (F5). On the staircase calibration window it scores 98.6%.
- **`w ∝ R⁻¹h`** was underspecified: normalization, covariance sampling, and
  the calibrated detector must be tested together. The subtraction proposal
  failed; self-cancellation as Stage B's failure cause remains unconfirmed (F1).

## Recommended order of work

The sub-01 diagnosis is complete on its three existing caches; two missing
conditions were skipped without preparing new data.

1. Prioritize spike likelihood (83.51% on existing outer records) and template
   tracking, now that posture targets have measured reliability. These remain
   research priorities; they do not independently confirm generalization or
   justify retraining on a noisy-target argument.
2. Consider out-of-sample Stage B only as an optional improvement. Synthetic
   ramp gains are 1.27 and 1.62 percentage points. Do not evaluate this change or
   a corrected robust decoder on sub-02 through sub-05.

Future confirmation needs genuinely new records (F9).

## Files changed by this review

- `README.md`: residual kinematics; phase-window warning.
- `benchmarks/pareto_residual/README.md`: kinematic inputs declared.
- `benchmarks/robust_mvdr/`: robust filter, inference, evaluation, audits, CLI,
  tests and the synthetic study. No existing hashed file was modified, so the
  provenance of earlier experiments is intact.
- `docs/benchmarks/robust_mvdr_synthetic_*.json`: synthetic study outputs.

Follow-up additions: `benchmarks/fifo_audit/`,
`docs/benchmarks/fifo_audit_synthetic.json`,
`docs/benchmarks/fifo_audit_sub01.json`, and
`docs/benchmarks/robust_mvdr_correction.md`. These preserve the original
registered experiment and record the withdrawn proposal's diagnostic limits.
