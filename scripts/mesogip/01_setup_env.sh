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

ADAPT_DIR="$REPO_ROOT/external/farina/adapt_decomp"
ADAPT_COMMIT="a8f53736e297962670276bc123c137050b7056b6"
if [ ! -d "$ADAPT_DIR/.git" ]; then
    mkdir -p "$REPO_ROOT/external/farina"
    git clone https://github.com/imendezguerra/adapt_decomp.git "$ADAPT_DIR"
fi
git -C "$ADAPT_DIR" checkout "$ADAPT_COMMIT"
python -m pip install --no-deps -e "$ADAPT_DIR"

python -c "import torch, numpy, scipy, optuna; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo "=== Setup complete ==="
