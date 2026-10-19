# Independent review of the movement-robust decomposition work — 19 October 2026

Follow-up: the [CPU diagnostic](robust_mvdr_diagnostic.md) records the completed FDSI experiment, the measured target-reliability audit and the fixes applied. The findings below preserve the original review; its self-cancellation and target-noise diagnoses must be assessed against those measurements.

The FDSI follow-up measured only 5.21–5.30% RoA for the six proposed variants,
against 45.56% for the original static analytic decoder. The FIFO control also
collapsed. A source-reading correction matters: Stage B divides by
`sqrt(h^T R^-1 h)`, whereas the proposed unit-response MVDR divides by
`h^T R^-1 h`. Equal filter direction does not imply equal amplitude or an
identical decoder. The cause of the shared collapse has not been isolated;
the original F1 diagnosis is therefore not established by this run. The
target-reliability ratio was 30.9224, rather than near one, so the F2 claim of
noise-dominated posture targets is also unsupported by this audit. No numerical
code was changed after registration and the negative results remain intact.

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
- FDSI data were not available in the review environment, so no new FDSI
  prediction was run. The proposed experiment is registered as code and is
  ready to run on CPU.

## Verdict by stage

| Stage | Implementation | Conclusion | Main issue |
| --- | --- | --- | --- |
| Farina reproduction | Correct | Valid | None |
| Stage A | Correct | Valid no-go | Design: rank-3 correction cannot represent the drift (F3) |
| Pareto residual | Correct | Valid, but mislabelled | Uses measured angle and velocity, so it is K-type (F4) |
| Template rescue | Correct | Valid no-go | `F(H)` ignores uncalibrated interference (F6) |
| Stage B MoCA | Correct | No-go valid; diagnosis incomplete | MVDR self-cancellation (F1); Γ targets and metric (F2) |
| Separation/geometry pilots | Correct | Valid stop | Fixed the metric, not the self-cancellation (F1) |
| Spike likelihood | Correct, no leakage | Valid | Gain is mostly added candidates (F7) |
| Phase reporting, all stages | Reproduces the reference | Misleading for ramps | "First hold" contains movement (F5) |

## Findings

### F1 — Stage B's low recall is MVDR self-cancellation (major, design)

`w = R⁻¹h / (hᵀR⁻¹h)` is a Capon/MVDR beamformer, and the FIFO covariance `R`
contains the target unit. With a mismatched `h`, MVDR partially cancels the true unit
([Cox, 1973](https://doi.org/10.1121/1.1913659);
[Li, Stoica and Wang, 2003](https://doi.org/10.1109/TSP.2003.812831)). Its response
falls under the threshold calibrated at rest while other units stay suppressed.
Stage B shows exactly that signature:

- Precision is about 92% for every analytic variant, while recall is 39–48%,
  against 83% for Pareto.
- In the final hold, where the static template is correct again, the analytic
  decoder beats Pareto: 97.50% against 87.99%.
- The learned morph (MoCA-K 36.97%) is worse than no morph (static 45.56%).

Synthetic test, with a 10 s hold at a fixed shortened length (no covariance lag):

| s₁ | Static template, full `R`, fixed threshold | Same, target removed from `R` (oracle) | Best threshold, full `R` |
| --- | ---: | ---: | ---: |
| 0.96 | 0.737 | 1.000 | 0.975 |
| 0.94 | 0.569 | 0.997 | 0.926 |
| 0.91 | 0.311 | 0.870 | 0.862 |

The information is still there (high best-threshold RoA). The fixed-threshold loss
comes from response amplitude, and it disappears when the target is excluded from
`R`. Practical remedies were compared without labels. The partial model subtraction
`R − 0.5·rate·CᵀC` with 0.03 diagonal loading was best, or within 0.3 points of
best, with an unsupervised two-means threshold in all nine cells. At s₁ = 0.94 with
the static template it raised fixed-threshold RoA from 0.585 to 0.848. Detection
blanking helped little (0.677). Stage B's α = 0.5 model mixing costs about 25 points
at a fixed threshold even with an exact template (0.74–0.76 against 0.99–1.00). Recall
stays at 0.99, so the loss is false positives from uncalibrated units it no longer
suppresses.

The pilots also support this reading. With an estimated ground-truth-triggered
template, the full-`R` MVDR reached 96.25% on the `sub-01` staircase but only
78.87% (α = 0.5) and 80.26% (α = 0) across `sub-01` conditions. That is Pareto level,
not above it.

**Change:** remove self-cancellation before judging any morph. Implemented, not yet
run, in [`benchmarks/robust_mvdr`](../../benchmarks/robust_mvdr/README.md).

### F2 — Γ was trained on noisy targets with the wrong metric (major)

The morph Γ was trained with NMSE on raw MUAP shape. The separator, however, weights
errors by `R⁻¹`, which amplifies exactly the low-variance directions that raw NMSE
ignores. The separation pilot fixed the metric, but the identity still beat Γ.
The geometry pilot's identity error already exceeds 100% in whitened geometry.
That suggests the spike-triggered targets are dominated by noise in the directions
that matter. Their noise ceiling was never measured.

**Change:** before training another morph, run
`python -m benchmarks.robust_mvdr.run target-reliability`. It uses `sub-01` only and
reports the split-half noise of the posture-binned targets against their change
from calibration, both raw and in the separation geometry. If the ratio is near one,
those targets cannot teach Γ anything.

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
the 0–10 s window likely contains movement. Earlier phase comparisons ("MoCA loses
15 points in the first hold") are therefore early-movement comparisons.

**Change:** `phase-audit` checks this from the angle traces alone. The robust
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
weak candidates. Precision falls slightly, from 91.47% to 91.10%. That fits F1: the
missing discharges are under-threshold events, not absent ones.

### F8 — CI does not lint or test `benchmarks/` (minor)

`.github/workflows/validate.yml` runs Ruff and pytest on `src`, `scripts` and
`tests` only. The benchmark suites run only by hand.

**Change:** add a CPU job with `benchmarks/stage_a/requirements.lock.txt` and the
pinned `adapt_decomp` checkout.

### F9 — The outer subjects are no longer a fresh test set (moderate, protocol)

`sub-02`–`sub-05` have been scored by five experiments. Each one was registered
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
- **`w ∝ R⁻¹h` with a model-shrunk covariance** was underspecified. Without
  removing the target or adding loading it self-cancels (F1).

## Recommended order of work

1. Run `phase-audit` and `target-reliability`; they are seconds to minutes on CPU.
2. Run the pre-registered robust decoding experiment (`register` → `predict` ×4 →
   `seal` → `score` → `report`). It takes about 2–4 h on CPU and trains nothing.
3. If `static_subtract` improves recall, add a label-free adaptive threshold, or
   apply the spike-likelihood layer on top of it.
4. Retrain a morph only if target reliability shows a learnable signal. Train it in
   the separation geometry, with the robust covariance in the loop.
5. Use the static analytic decoder near the reference posture, where it already
   beats Pareto, and confirm everything on new data (F9).
6. Add the benchmark CI job (F8).

## Files changed by this review

- `README.md`: residual kinematics; phase-window warning.
- `benchmarks/pareto_residual/README.md`: kinematic inputs declared.
- `benchmarks/robust_mvdr/`: robust filter, inference, evaluation, audits, CLI,
  tests and the synthetic study. No existing hashed file was modified, so the
  provenance of earlier experiments is intact.
- `docs/benchmarks/robust_mvdr_synthetic_*.json`: synthetic study outputs.
