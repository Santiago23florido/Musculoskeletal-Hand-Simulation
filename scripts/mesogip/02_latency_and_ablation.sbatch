#!/bin/bash
#SBATCH --job-name=embc-p3-diag
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:50:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# EMBC 2027 Phase 1 job on Mesogip:
#   1. Hardware + Apptainer/NeuroMotion probe
#   2. Clean Linux latency breakdown (CPU 1-thread, CPU 4-thread, and L40S CUDA head)
#   3. Zero-retraining P3 calibration-memory ablation (sub-04, frozen seeds 0-2, resumable)

set -euo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${MESOGIP_VENV:-$REPO_ROOT/.venv-mesogip}"

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"
export PARETO_MC_OUTPUT="${PARETO_MC_OUTPUT:-outputs/pareto_mc/development_v2}"
export OPENBLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

trap 'echo "[WARN] Approaching 4h walltime limit (SIGUSR1) at $(date -u). Completed records are saved and resumable."' USR1

echo "--- [1/4] Cluster & NeuroMotion runtime probe ---"
python -m benchmarks.pareto_mc.embc_cluster probe

echo "--- [2/4] Clean Linux latency profile (1 thread + CUDA head) ---"
python -m benchmarks.pareto_mc.embc_cluster profile-latency --threads 1 --head-device cuda

echo "--- [3/4] Clean Linux latency profile (4 threads CPU) ---"
python -m benchmarks.pareto_mc.embc_cluster profile-latency --threads 4 --head-device cpu

echo "--- [4/4] Zero-retraining P3 memory ablation on sub-04 (resumable) ---"
python -m benchmarks.pareto_mc.embc_cluster ablate-memory --workers 4 --threads 2

echo "=== Completed at $(date -u) ==="
