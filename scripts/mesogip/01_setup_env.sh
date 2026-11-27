#!/usr/bin/env bash
# Set up the Linux CUDA environment on Mesogip for Pareto-MC benchmarks.
# Run from the repository root on Mesogip (or inside a short interactive srun).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${MESOGIP_VENV:-$REPO_ROOT/.venv-mesogip}"

mkdir -p "$REPO_ROOT/logs"

echo "=== Setting up Mesogip Python 3.12 environment in $VENV_DIR ==="
if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    grep -v -iE '^(win32-setctime|torch==)' "$REPO_ROOT/benchmarks/stage_b/requirements-gpu.lock.txt" > /tmp/mesogip_reqs.txt
    uv pip install -r /tmp/mesogip_reqs.txt
    uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
else
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    grep -v -iE '^(win32-setctime|torch==)' "$REPO_ROOT/benchmarks/stage_b/requirements-gpu.lock.txt" > /tmp/mesogip_reqs.txt
    python -m pip install -r /tmp/mesogip_reqs.txt
    python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
fi

if [ -d "$REPO_ROOT/external/farina/adapt_decomp" ]; then
    python -m pip install --no-deps -e "$REPO_ROOT/external/farina/adapt_decomp"
else
    echo "[WARN] external/farina/adapt_decomp not found yet. Sync external/ or run git clone for adapt_decomp."
fi

python -c "import torch, numpy, scipy, optuna; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo "=== Setup complete ==="
