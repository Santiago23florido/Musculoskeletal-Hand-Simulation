# Publication validation - 19 October 2026

This record covers the source snapshot prepared for GitHub publication on
19 October 2026. It preserves earlier experimental summaries without rerunning
training, downloading the full archives again, or regenerating their reported
scientific results.

## Local checks

| Check | Result |
| --- | --- |
| Physical simulation suite, native Windows environment | 46 passed, 1 graphics test skipped |
| Benchmark suite, CPU environment | 75 passed, 2 CUDA tests skipped |
| Benchmark suite, GPU environment | 77 passed |
| Ruff lint and formatting across `src`, `scripts`, `tests`, and `benchmarks` | Passed |
| Native simulation smoke run | 200 steps, 0.4 simulated seconds, finite actuator forces |
| Model inventory export | 23 joints, 39 actuators/tendons, 496 sites, zero registered sensors |
| Optional research-guide provenance | 12 registration checks passed across six workflows |

Commands used from the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-windows'
uv run --locked pytest -q
uv run --locked ruff check src scripts tests benchmarks
uv run --locked ruff format --check src scripts tests benchmarks
uv run --locked python scripts/run_simulation.py --steps 200 --output outputs/publication-smoke.npz
uv run --locked python scripts/inspect_model.py --output outputs/publication-inventory.json
.\.venv-benchmarks\Scripts\python.exe -m pytest -q benchmarks --basetemp=.pytest_tmp_publish_benchmarks
.\.venv-stage-b-gpu\Scripts\python.exe -m pytest -q benchmarks --basetemp=.pytest_tmp_publish_gpu
```

## Publication changes

PDF guides, agent instruction files, local assistant configuration, and the
internal `lessons/` journal are excluded by `.gitignore`. Public documentation
links point to tracked benchmark summaries and public experiment READMEs.
Downloaded source code, datasets, virtual environments, and runtime artifacts
remain ignored.

Research-guide hashes are optional when creating new run metadata. Registration
was exercised for the Pareto residual, template rescue, Stage B, separation,
geometry, and spike-likelihood workflows, both with the original guide available
and with its absence simulated. Each available-guide hash matched the file's
SHA-256; each absent-guide hash was JSON `null`. Source/specification checks
passed for every newly registered protocol. Tests used fresh ignored output
directories and left previous runs intact.

Archived JSON/CSV results retain their original values and provenance. Publication
changes to optional guide metadata alter source hashes, so old frozen output
directories must remain preserved; new experiments require fresh directories.
These checks validate the published implementation and its packaging. They do
not establish independent-generator transfer or full live-stream performance.
