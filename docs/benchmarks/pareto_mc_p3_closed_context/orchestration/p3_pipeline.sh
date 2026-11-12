#!/usr/bin/env bash
# P3, unattended (relaunch after the label-cache race): driver for rounds/final/analysis/
# latency, then the single registered opening of sub-05. Steps are idempotent.
set -u
cd "$(git rev-parse --show-toplevel)"
export PARETO_MC_OUTPUT=outputs/pareto_mc/development_v2 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE P3_WAIT_SEEDS=0
PY=.venv-benchmarks/Scripts/python.exe
echo "== P3 run start $(date -u +%FT%TZ)"
$PY -u outputs/pareto_mc/development_v2/p3_driver.py 4 || { echo "P3 FAILED run $(date -u +%FT%TZ)"; exit 1; }
echo "== P3 heldout start $(date -u +%FT%TZ)"
$PY -u -m benchmarks.pareto_mc.closed_context heldout --workers 4 || { echo "P3 FAILED heldout $(date -u +%FT%TZ)"; exit 1; }
echo "P3 COMPLETE $(date -u +%FT%TZ)"
