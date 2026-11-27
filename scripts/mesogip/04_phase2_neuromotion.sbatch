#!/bin/bash
#SBATCH --job-name=embc-p2-nm
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:50:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# EMBC 2027 Phase 2 job on Mesogip:
#   Confirmatory unseen subjects (seeds 210601-210603) + OOD movements (ramp3s, sinusoid)
#   generated via official MUniverse NeuroMotion Singularity container on L40S GPU,
#   followed by 5-s FDSI CBSS calibration and frozen Pareto/L2/P3/P3-nomem evaluation.

set -euo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${MESOGIP_VENV:-$HOME/venvs/pareto-mc}"
SIF_DIR="${SIF_DIR:-/tmp/ensta_santiago_sif}"
SIF_PATH="$SIF_DIR/muniverse_neuromotion.sif"

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

# Ensure pinned MUniverse repo is present for _run_neuromotion.py
MUNIVERSE_DIR="$REPO_ROOT/external/pareto_mc_muniverse"
if [ ! -d "$MUNIVERSE_DIR/.git" ]; then
    mkdir -p "$REPO_ROOT/external"
    git clone https://github.com/dfarinagroup/muniverse.git "$MUNIVERSE_DIR"
    git -C "$MUNIVERSE_DIR" checkout 94fbfab253f037e3b3cbfdbc6aed85b781f0ea49
fi

# Pull Singularity image into node-local /tmp (keeps shared NFS $HOME clean!)
mkdir -p "$SIF_DIR"
export SINGULARITY_CACHEDIR="$SIF_DIR/cache"
export SINGULARITY_TMPDIR="$SIF_DIR/tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

if [ ! -f "$SIF_PATH" ]; then
    echo "--- Pulling official NeuroMotion container into node-local /tmp ---"
    singularity pull "$SIF_PATH" docker://pranavm19/muniverse:neuromotion
fi

trap 'echo "[WARN] Approaching 4h walltime limit (SIGUSR1) at $(date -u). All generated/calibrated/evaluated files are resumable."' USR1

echo "--- [1/3] Generating confirmatory & OOD recordings ---"
python -m benchmarks.pareto_mc.embc_neuromotion generate --sif "$SIF_PATH"

echo "--- [2/3] Running 5-s FDSI CBSS calibrations ---"
python -m benchmarks.pareto_mc.embc_neuromotion calibrate --workers 4

echo "--- [3/3] Evaluating frozen Pareto / L2 / P3 / P3-nomem ---"
python -m benchmarks.pareto_mc.embc_neuromotion evaluate --workers 4 --threads 2

# Clean node-local container image to leave node /tmp spotless
rm -rf "$SIF_DIR"

echo "=== Phase 2 finished at $(date -u) ==="
