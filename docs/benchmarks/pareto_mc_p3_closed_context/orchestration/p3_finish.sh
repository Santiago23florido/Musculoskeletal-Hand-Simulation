#!/usr/bin/env bash
# P3 finish (11 Nov 2026, relaunch): (1) registered latency, four threads, default OpenMP
# wait policy, run overnight for isolation; (2) post hoc one-thread latency (oneMKL LU
# pathology with four threads on this laptop); (3) the single registered opening of
# sub-05 with the development settings (4 workers x 4 threads). Registered module unchanged.
set -u
cd "$(git rev-parse --show-toplevel)"
export PARETO_MC_OUTPUT=outputs/pareto_mc/development_v2 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset KMP_BLOCKTIME OMP_WAIT_POLICY
PY=.venv-benchmarks/Scripts/python.exe
echo "== P3 latency start $(date -u +%FT%TZ)"
$PY -u -m benchmarks.pareto_mc.closed_context latency || { echo "P3 FAILED latency $(date -u +%FT%TZ)"; exit 1; }
echo "== P3 latency threads1 start $(date -u +%FT%TZ)"
$PY -u outputs/pareto_mc/development_v2/p3_latency_threads1.py || { echo "P3 FAILED latency1 $(date -u +%FT%TZ)"; exit 1; }
echo "== P3 heldout start $(date -u +%FT%TZ)"
export KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE
$PY -u -m benchmarks.pareto_mc.closed_context heldout --workers 4 || { echo "P3 FAILED heldout $(date -u +%FT%TZ)"; exit 1; }
echo "P3 COMPLETE $(date -u +%FT%TZ)"
