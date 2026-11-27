<#
.SYNOPSIS
    Sync only the required FDSI data, pinned adapt_decomp, and frozen P3/L2/C1 checkpoints
    from Windows to Mesogip (skips the 45 GB training caches).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/mesogip/03_sync_to_cluster.ps1 -RemoteRepo "~/proyectos/Musculoskeletal-Hand-Simulation"
#>
param(
    [string]$HostAlias = "mesogip",
    [string]$RemoteRepo = "~/proyectos/Musculoskeletal-Hand-Simulation"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking SSH connectivity to $HostAlias..." -ForegroundColor Cyan
ssh $HostAlias "mkdir -p $RemoteRepo/data/benchmarks/farina $RemoteRepo/external/farina $RemoteRepo/outputs/pareto_mc/development_v2"

$paths = @(
    @{ Local = "data/benchmarks/farina/fdsi_benchmark/data/sub-04"; Remote = "$RemoteRepo/data/benchmarks/farina/fdsi_benchmark/data/" },
    @{ Local = "data/benchmarks/farina/fdsi_benchmark/outputs/calibration/sub-04"; Remote = "$RemoteRepo/data/benchmarks/farina/fdsi_benchmark/outputs/calibration/" },
    @{ Local = "data/benchmarks/farina/fdsi_benchmark/outputs/adaptation/sub-04/lr_fixed/tpe_pareto/*_config.yaml"; Remote = "$RemoteRepo/data/benchmarks/farina/fdsi_benchmark/outputs/adaptation/sub-04/lr_fixed/tpe_pareto/" },
    @{ Local = "outputs/pareto_mc/development_v2/C1/seed0"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/C1/" },
    @{ Local = "outputs/pareto_mc/development_v2/L0/validation/scores"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/L0/validation/" },
    @{ Local = "outputs/pareto_mc/development_v2/L0/validation/summary.json"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/L0/validation/" },
    @{ Local = "outputs/pareto_mc/development_v2/L2/retained.json"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/L2/" },
    @{ Local = "outputs/pareto_mc/development_v2/L2/standard/round3/seed0"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/L2/standard/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/L2/standard/round3/seed1"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/L2/standard/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/L2/standard/round3/seed2"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/L2/standard/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/round3/selected.json"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/round3/seed0"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/round3/seed1"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/round3/seed2"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/round3/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/final_validation/seed0/scores"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/final_validation/seed0/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/final_validation/seed1/scores"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/final_validation/seed1/" },
    @{ Local = "outputs/pareto_mc/development_v2/P3_closed_context/final_validation/seed2/scores"; Remote = "$RemoteRepo/outputs/pareto_mc/development_v2/P3_closed_context/final_validation/seed2/" }
)

$hasRsync = $null -ne (Get-Command rsync -ErrorAction SilentlyContinue)
foreach ($item in $paths) {
    if (-not (Test-Path $item.Local)) {
        Write-Warning "Skipping missing local path: $($item.Local)"
        continue
    }
    Write-Host "Syncing $($item.Local) -> ${HostAlias}:$($item.Remote)" -ForegroundColor Green
    ssh $HostAlias "mkdir -p $($item.Remote)"
    if ($hasRsync) {
        rsync -avhP $item.Local "${HostAlias}:$($item.Remote)"
    } else {
        scp -r $item.Local "${HostAlias}:$($item.Remote)"
    }
}

Write-Host "Sync complete! Next on Mesogip: bash scripts/mesogip/01_setup_env.sh && sbatch scripts/mesogip/02_latency_and_ablation.sbatch" -ForegroundColor Cyan
