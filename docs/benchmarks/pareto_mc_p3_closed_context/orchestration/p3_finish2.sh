#!/usr/bin/env bash
# P3 finish, relaunch after the Windows Update reboot of 12 Nov 2026 00:30 UTC.
# (1) the single registered opening of sub-05 (development settings: 4 workers x 4
# threads); (2) post hoc one-thread latency. latency.json (registered) already exists.
set -u
cd "$(git rev-parse --show-toplevel)"
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 PARETO_MC_OUTPUT=outputs/pareto_mc/development_v2 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv-benchmarks/Scripts/python.exe
echo "== P3 heldout start $(date -u +%FT%TZ)"
KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE $PY -u -m benchmarks.pareto_mc.closed_context heldout --workers 4 || { echo "P3 FAILED heldout $(date -u +%FT%TZ)"; exit 1; }
echo "== P3 latency threads1 start $(date -u +%FT%TZ)"
$PY -u outputs/pareto_mc/development_v2/p3_latency_threads1.py || { echo "P3 FAILED latency1 $(date -u +%FT%TZ)"; exit 1; }
echo "P3 COMPLETE $(date -u +%FT%TZ)"
