# Pareto-MC

Pareto adaptation with EMG-only state estimation, feedback from discharge
decisions, and memory local to each recording. **L2 is the latest accepted
development stage: 91.66% global RoA and 85.05% on 5-second ramps.** These are
validation results; external evaluation and the final latency target remain
pending. **L3 is paused at the user's request.**

## Development protocol

Development started from `review/robust-mvdr` while PR #1 was open. Training uses
sub-01–03, validation uses sub-04, and sub-05 is reserved. **The change from
Stage B was declared before training:** all conditions and all four SNR levels
are allowed during development. Sub-05 is excluded from tuning and diagnostics.
The [journal](../../docs/benchmarks/pareto_mc_journal.md) records decisions and
deviations.

Phase 0 attempted to start the official NeuroMotion backend from
[MUniverse](https://github.com/dfarinagroup/muniverse), revision
`94fbfab253f037e3b3cbfdbc6aed85b781f0ea49`. Docker's Linux engine was inactive,
and Ubuntu had neither Singularity nor Apptainer. No confirmatory recordings
were generated, so the study remains exploratory. Commands, errors, and hashes
of the Dockerfile and launcher are retained in `phase0.json`.

## Method and limits

The backbone retains the operations of
[`adapt_decomp` a8f5373](https://github.com/imendezguerra/adapt_decomp/tree/a8f53736e297962670276bc123c137050b7056b6),
using
[`optim_muniverse_fdsi_pareto.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/a8f53736e297962670276bc123c137050b7056b6/configs/adapt_configs/optim_muniverse_fdsi_pareto.yaml):
the fixed whitening step `W -= eta_wh (Rz-I) W`, reference FIFO, shrinkage and
clipping, wh→sv coordinate correction, `tanh(y)` contrast only at reliable
discharges, QR, and EMA centroids (0.95). The learning rates are 0.04038249 and
0.002422225. The implementation retains the safeguards omitted from this
abbreviated formula.

This can be interpreted as self-trained tracking of whitened templates. The
trade-off between drift tracking and noise in adaptive filters motivates
state-dependent steps, without guaranteeing a universally optimal step.
See [Widrow et al., 1975](https://doi.org/10.1109/PROC.1975.10036) and
[Kalman, 1960](https://doi.org/10.1115/1.3662552).
In ramp5s, posture repeats approximately eight times. Local memory aims to reuse
filters learned from the recording itself. The method does not assume that
cross-subject morphing with Γ succeeded.

The [FIFO audit](../fifo_audit/README.md) shows the risk of an abrupt amplitude
loss relative to stateful centroids. Mixing is bounded **after QR**; `beta=0`
skips every additional filter operation. Memory stores effective filters
`v=W.T@w` and centroids, with EMA per bin and confidence combining rate,
separation, and IQR. Retrieval uses weighted kernel regression
([Nadaraya, 1964](https://doi.org/10.1137/1109020)). The current batch is inserted
only after retrieval, and memory is never shared between recordings.

The earlier [spike-likelihood improvement](../spike_likelihood/README.md)
motivates learning discharge decisions. Feeding decisions back changes the
training observations; the aggregated rounds follow the principle of
[DAgger, Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html).
This motivates the procedure, without establishing the paper's bounds for EMG.

C1 retains GRU(120,24), adds mean and log-variance outputs, and uses Gaussian NLL.
C3 has 159 parameters and bounded outputs; multipliers start at one and beta at
zero. C4 extends the 19→8→1 MLP to 22→8→1 (193 parameters), adding source,
confidence, and estimated velocity. Its final layer starts at zero, so the
untrained decisions and adaptation reproduce Pareto.

## C0 equivalence check

On three real sub-01 records (staircase/ramp10s/ramp5s, 30 dB), **all samples
match, with zero RoA differences from both the original loop and the published
outputs**. The protocol retains the conservative name **Pareto-rerun**. Batch
medians are 38.05/38.48/38.12 ms and p95 values are 46.70/47.21/46.55 ms.
The 22 ms target is unmet with four threads on this machine. These three
training records establish equivalence; their averages are not an external
80-record evaluation.

The operational interface receives raw EMG and a sequential index. Filtering
uses stateful `sosfilt`, and extension uses past samples only. Reference filtering
is also causal: “offline” describes preparation of the recording, rather than
`filtfilt`. The new interface avoids the global mean used by `Data`. Outputs can
be delayed by up to one batch (100 ms); the supplied initial five-second
calibration is allowed. True angles and discharges are opened only in separate
supervision/evaluation functions with subject guards.

## Reproduction

Each run uses a fresh directory to preserve protocols, sealed predictions, and
results. The first L0 attempt in `outputs/pareto_mc/L0` failed because of a
constructor signature mismatch and is retained.

```powershell
$env:PARETO_MC_OUTPUT = 'outputs/pareto_mc/reproduction'
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.phase0
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.run identity --threads 4
.venv-stage-b-gpu/Scripts/python.exe -m benchmarks.pareto_mc.run train-observer --threads 4
$env:OPENBLAS_NUM_THREADS = '1'
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.run identity-head --threads 4
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.run l1 --threads 4
.venv-benchmarks/Scripts/python.exe -m pytest benchmarks/pareto_mc -q
.venv-windows/Scripts/python.exe -m ruff check benchmarks/pareto_mc
.venv-windows/Scripts/python.exe -m ruff format --check benchmarks/pareto_mc
```

Training uses `.venv-stage-b-gpu` on the RTX 4070; inference and timing use
`.venv-benchmarks` on CPU. The original constructor emits Intel MKL `SLASWP`
errors when estimating batch KL statistics on Windows. Those statistics do not
scale the steps under `lr_mode=fixed`. Exact agreement is checked on all three
controls despite those messages. The pinned dependency is preserved.

Per-stage diagnostics include true amplitude through `amplitude_metrics`,
centroids, memory hits, control versus estimated velocity, units with recall
below 0.5, and timing distributions. Phases use `angle_phases`. Cycles are
angular excursions above 1° that reach at least 35°, excluding calibration;
they are not constructed from fixed time intervals.

## Development run

**C1 completed 30 epochs**, using seed 0 and selecting the checkpoint at epoch 13
by sub-04 NLL: −1.87788, with a descriptive angular RMSE of 3.2407°. Weights are
in `outputs/pareto_mc/development_v2/C1/seed0/observer.pt`, alongside the original
manifest. Measured angles occur only in the training/evaluation target.

The first L1 run (`development_v1`) is retained. It was superseded before
RoA/Jaccard selection. `development_v2` reuses inputs and a completed checkpoint,
verified in `input_reuse.json`. The training cache preserves the exact arrays
and labels per record, avoiding repeated lag estimation and decompression at
each epoch. Files and provenance have hashes and subject checks.

Torch uses four threads, as in the exact C0 control. Two threads produced seven
differences in one record due to BLAS numerical differences. With four threads,
**untrained C4 again produces zero differences on all three real records**,
including with the trained observer and a single window for KL statistics,
which fixed-step adaptation does not use. Timing under contention in these
checks is separate from an isolated final latency measurement.

L1 aggregates rounds 0–3, with three seeds per round. The checkpoint is selected
by validation BCE. Then 40 TPE trials tune threshold and residual strength by
Jaccard with calibration lags. The best three candidates are repeated with
three seeds and selected by the median. Memory parameters remain inactive
until L2. The acceptance check runs after full-model inference, using the
original evaluator and previously sealed predictions.

If full feedback fails the criteria, one structural alternative is
preregistered: added events may update SV/centroids only when their posterior
is at least 0.8; accepted events remain predictions. This threshold is reachable
with the residual bounded to four; 0.9 was unreachable for new candidates.
If the alternative also fails, L0 is retained and L2–L4 are not run.

Pareto is retained during calibration for batches starting before 5 s, which can
include the remaining fraction of the last batch (about 80 ms). Those timestamps
are not inference labels. The open-loop reference retains exactly the first
five seconds of original predictions.

### L1 completed

**The first structure passes the L1 criteria**, so the alternative was not run.
Sub-04 validation covers 20 records and 124 instances, scored with the original
evaluator:

| Stage | Global RoA | ramp5s | ≥90% | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Pareto-rerun | 87.65% | 73.88% | 66.94% | 96.06% | 89.03% |
| Open-loop C4 | 90.39% | 81.77% | 70.16% | 95.70% | 92.94% |
| Closed-loop L1 | **91.24%** | **84.71%** | **72.58%** | 96.38% | 93.01% |

**L1 gains 0.84 percentage points globally and 2.94 points on ramp5s over
open-loop C4.** Reference-rest RoA is 97.13%, moving RoA 90.48%, and
away-from-reference plateau RoA 89.98%. The first ramp5s cycle scores 65.89%;
cycles 2–8 score 84.94%. With four concurrent runs, measured latency was
127.92 ms median and 151.32 ms p95. This exceeds 22 ms and is not an isolated
measurement. Results are exploratory development validation, with neither the
external 80-record evaluation nor confirmation on new data completed.

Details, acceptance checks, limits, and hashes are in the
[journal](../../docs/benchmarks/pareto_mc_journal.md) and
[versioned summary](../../docs/benchmarks/pareto_mc_l1/pareto_mc_summary.json).

### L2 method

`memory_loop.py` adds causal retrieval of effective filters and their centroids,
projection into the current whitening coordinates, uncertainty attenuation, and
bounded mixing after QR. In L2, only the calibration snapshot is available.
Zero beta exactly reproduces Pareto on the three real controls and preserves
all L1 state in integration tests.

```powershell
# Requires completed L0, C1, and L1 stages in PARETO_MC_OUTPUT.
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.memory_identity "$env:PARETO_MC_OUTPUT/C2_identity_v1"
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.shared_identity
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.memory_experiment L2 --workers 4
```

The search registers 45 trials, five per structural combination, then repeats
the best three with all three trained C4 seeds. It shares only preprocessing,
whitening, and C1 from the same batch across nine trajectories. A real-data
test requires bitwise equality with independent execution. Search timing is
not treated as inference latency. C4 is then retrained through three DAgger
rounds, and predictions are regenerated with the full CPU loop before checking
the criteria. All records are sealed before their labels are opened.

Bin momentum stays fixed in L2 because there are no insertions; it is searched
in L3. The limiter bounds the memory residual relative to the Pareto proposal.
Retrieved centroids are preserved, while live centroids retain their EMA update.
The journal records these choices and the single permitted alternative after
a failure. The L2 result is established by its saved `gate.json`.

An additional L1 profile with one record at a time measured 56.12 ms median
and 360.87 ms p95, also above 22 ms. Initial C2 tests and measurements are in
`docs/benchmarks/pareto_mc_l2_setup/`.

### L2 completed

**L2 passes its validation criteria with the first structure**, after 45 trials,
three-seed repetitions, and three new DAgger rounds. Sub-04 validation covers
20 records and 124 instances:

| Metric | L1 | L2 |
|---|---:|---:|
| Global RoA | 91.24% | **91.66%** |
| ramp5s RoA | 84.71% | **85.05%** |
| Instances ≥90% | 72.58% | **75.81%** |
| Precision | 96.38% | 95.68% |
| Recall | 93.01% | **94.14%** |
| Reference rest | 97.13% | **97.54%** |
| Moving | 90.48% | **90.94%** |

**L2 meets the reference-rest target of at least 97% and improves movement
accuracy over L1.** Precision is 0.38 percentage points below Pareto, within the
one-point allowance. Ramp5s scores 68.97% on cycle 1 and 85.01% on cycles 2–8.
Memory is active in 4.29% of unit/batch pairs.

CPU timing with four concurrent records is **86.16 ms median and 105.51 ms p95**,
above the 22 ms target. An isolated L2 profile remains pending. **L2 is retained
as the latest accepted development stage.** Results remain exploratory
validation, pending external evaluation and confirmation on new data. The
[versioned result](../../docs/benchmarks/pareto_mc_l2_standard/gate.json) records
the acceptance check and hashes.

### L3 setup and pause

L3 enables reliable insertions into memory bins and keeps calibration immutable.
It starts from the three final L2 checkpoints and adds three DAgger rounds,
also aggregating L1 and L2 trajectories. Bin momentum is now included in TPE.
Before the search, a real-data test requires bitwise equality between independent
and shared execution, including learned bins and their confidence.

```powershell
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.full_memory_identity "$env:PARETO_MC_OUTPUT/C2_full_memory_identity_v1"
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.full_memory_experiment --workers 4
```

The L3 criteria require global RoA of at least 92.6580%, and ramp5s cycles 2–8
at least five points above cycle 1. If the first structure fails, only the
registered local-confidence alternative is allowed. A second failure stops
progression to later stages.

**L3 has been paused at the user's request since 23 October 2026.** Its identity
checks passed, but the search stopped without scored trials. The study and
pending parameters are saved for an explicitly requested resumption. Publishing
or merging the code does not start experiments. L2 remains the latest accepted
stage; the [pause record](../../docs/benchmarks/pareto_mc_l3_setup/pause.json)
describes the saved state.
