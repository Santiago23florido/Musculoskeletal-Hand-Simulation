#!/usr/bin/env bash
# Run once on the Mesogip login node (ssh mesogip-tp) to inspect resources,
# storage quotas, Python/CUDA tools, and container runtimes (Apptainer/Singularity).
set -euo pipefail

echo "=== Host & User ==="
hostname && id && date -u

echo ""
echo "=== Slurm Partitions (sinfo -a) ==="
sinfo -a -o "%P %a %l %D %G %m %c" || true

echo ""
echo "=== Storage & Quotas ==="
df -h "$HOME" /tmp
quota -s 2>/dev/null || true
ls -ld /scratch /work /data /tmp 2>/dev/null || true

echo ""
echo "=== Python, UV, Conda, Containers ==="
for bin in python3 uv conda git rsync apptainer singularity docker; do
    if command -v "$bin" >/dev/null 2>&1; then
        printf "%-12s -> %s\n" "$bin" "$(command -v "$bin")"
    else
        printf "%-12s -> (not found)\n" "$bin"
    fi
done

echo ""
echo "=== Modules (if available) ==="
if command -v module >/dev/null 2>&1; then
    module avail 2>&1 | head -n 40 || true
else
    echo "No module command on login node."
fi
