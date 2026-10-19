# FIFO regime audit (CPU only)

This package diagnoses the withdrawn Robust MVDR proposal using the real static
decoder loops and detector. It does not modify the registered experiment, train
models, or evaluate the outer subjects again. See the
[measured correction](../../docs/benchmarks/robust_mvdr_correction.md).

From the repository root in PowerShell, use a fresh output directory:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.venv-benchmarks/Scripts/python.exe -m benchmarks.fifo_audit.harness --output outputs/fifo_audit_reproduction --threads 4
.venv-benchmarks/Scripts/python.exe -m benchmarks.fifo_audit.fdsi_check --subject sub-01 --output outputs/fifo_audit_reproduction --threads 4
```

The synthetic CLI renders four main scenarios and stationary sensitivity at
20/30 dB for seeds 3/4/5, with seven fixed variants. It records amplitude ratios,
paired RoA at 2 ms, per-unit tables, and the declared qualitative claims. Seed-3
stationary results are reused rather than counted as independent replicates.

The FDSI CLI accepts only sub-01. It opens existing 30 dB Stage B feature caches
without a cache-creation path; absent conditions are reported and skipped.
Real-predict controls determine the declared A/B/C diagnostic case. Case C stops
before ablations, saves the controls, and returns exit code 3. Replay mismatches
raise an assertion rather than silently changing the decoder. Output JSON files
are never overwritten.

The test suite checks frozen-predict equivalence, the D=512 stationary amplitude
loss, subject restrictions, and the missing-cache guard:

```powershell
.venv-benchmarks/Scripts/python.exe -m pytest benchmarks/fifo_audit -q
```

No changes should be made to frozen `benchmarks/robust_mvdr/*.py`, historical
Stage B data, or published `robust_mvdr_*.json`/CSV artifacts for this diagnosis.
