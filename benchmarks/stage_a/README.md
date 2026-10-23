# Stage A: learned whitening correction

An independent implementation of a learned whitening correction. The
[published summary](../../docs/benchmarks/stage_a_p1_summary.json) records the
results and the acceptance criterion for the P1 evaluation.

The architecture applies `W_b = (I + U(z_b)V(z_b)^T) W_0`, using six bases, rank 3,
a mixing MLP, and causal state. The certified spectral-norm bound is 0.8.
It retains the original calibrations, detection functions, IQR gate, centroids,
and separation updates from `adapt_decomp`, pinned to
`a8f53736e297962670276bc123c137050b7056b6`. This is an independent implementation;
the unavailable HarmonICA code could not be reproduced.

The main model uses the observed wrist angle (**Stage A-K**). It trains on
sequences with Gaussian KL and `log(cosh)` contrast consistency. Numerical
details are recorded in the modules and results linked below.

**The model did not meet the accuracy targets.** Five training runs and the full
P1 evaluation covered 100 records and 768 instances per method. Mean RoA was
57.83% for Stage A versus 79.25% for Pareto; differences on 5/10 s ramps and
staircase were −10.55/−20.74/−28.62 percentage points. The result is retained,
without selecting another model based on the test results.

## Reproduce in a fresh output directory

Run from the repository root in PowerShell. The full weeks 1–2 dataset must
already be available:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/farina/setup.ps1 -Fdsi
uv pip install --python .venv-benchmarks/Scripts/python.exe --requirement benchmarks/stage_a/requirements.lock.txt
$env:STAGE_A_OUTPUT = 'outputs/stage_a_reproduction'
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 6 prepare
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 audit
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train pilot_cov128 --epochs 3 --samples 128 --loss-mode delta --validation-whitening-weight 1 --validation-contrast-weight 0.01
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 6 train full_fifo_squared --epochs 8 --samples 2000 --loss-mode squared --validation-whitening-weight 1 --validation-contrast-weight 0.01
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train balanced_gru --epochs 10 --samples 2000 --whitening-weight 0.0001 --contrast-weight 0.1
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train balanced_mlp --epochs 10 --samples 2000 --encoder mlp --whitening-weight 0.0001 --contrast-weight 0.1
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train state3_gru --epochs 8 --samples 2000 --state-dim 3
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 select pilot_cov128 full_fifo_squared balanced_gru balanced_mlp state3_gru
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 evaluate outputs/stage_a_reproduction/selected.pt --name p1_complete
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.statistics --threads 2
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.report
Remove-Item Env:STAGE_A_OUTPUT
```

Training and selection names are protected against overwriting. Use a different
output directory for a new exploration. An evaluation can resume under the same
name when its plan, code, and checkpoint match; only completed records are
reused. The script does not select a different model based on the test results.

Training uses two ramps from the P1 pool (`sub-01`, 30 dB, 40 and 5 s), with
selection on the third ramp (10 s), exclusively by an unsupervised loss.
Supervised calibration on the first five seconds is retained. The shared
selection objective uses weights of 0.0001 for KL and 0.1 for contrast. True
discharges after calibration are opened only to evaluate saved predictions.
The first two training runs selected epochs with their original weights of
1/0.01, reproduced by the `--validation-...` options above. Final selection
rescores all five candidates with the shared objective.

The full evaluation is equivalent to five subject partitions. To distribute
the computation, run `evaluate ... --name p1_sub01 --subjects sub-01` and the
equivalent commands for the other four subjects, keeping the same checkpoint,
code, methods, and thread count. Then
`merge p1_sub01 p1_sub02 p1_sub03 p1_sub04 p1_sub05` creates
`evaluations/p1_complete/`. Merging rejects overlaps and incomplete coverage.
The recorded run used at most three processes with two threads per process.

## Files and results

- `data.py`: prepares signals and features without loading true discharges.
- `model.py`: encoder, invertible correction, and differentiable losses.
- `objectives.py`: FIFO KL constants and calibration statistics, using Cholesky.
- `train.py`, `selection.py`: sequences, checkpoints, and selection before testing.
- `inference.py`: original detector/SV with the new whitening and coupling.
- `evaluate.py`, `merge.py`: P1, controls, metrics, paired tests, and coverage.
- `statistics.py`: mixed model `method × condition + (1 | subject/record)`.
- `report.py`: figures regenerated from the computed records.

Outputs default to `outputs/stage_a/`, which Git ignores. Each training run
saves its protocol, seed, hashes, source snapshot, losses, and weights. Selection
saves the candidate hash. Each evaluation retains code, predictions, per-unit
scores, phases, and timing. Published Pareto results are rescored and must match
their saved RoA values within `1e-10`.

The acceptance criterion requires **a gain of at least 5 percentage points on
both 5 and 10 s ramps, with no staircase loss**, across all 768 instances of the
full P1 evaluation. A partial evaluation cannot establish acceptance. Inference
with cached `W_0 x` retains the offline protocol, so its timing does not establish
live-system latency.

## Verification

```powershell
.\.venv-benchmarks\Scripts\python.exe -m pytest -q benchmarks/stage_a/test_stage_a.py benchmarks/farina/test_metrics.py benchmarks/farina/test_fdsi.py benchmarks/farina/test_download.py
.\.venv-windows\Scripts\ruff.exe check benchmarks/stage_a
```

Tests check equivalence and gradients against dense computation, Cholesky,
invertibility, causality, the reference posture, Woodbury coupling, contrast
against the authors' implementation, and rejection of acceptance based on
partial data. Before training, the audit on real benchmark signals reproduces
the fixed comparator bit for bit.
