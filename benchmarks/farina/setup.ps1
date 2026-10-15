param([switch]$Fdsi)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $repoRoot
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'sources.json') -Raw | ConvertFrom-Json
$reference = Join-Path $repoRoot 'external/farina/adapt_decomp'
$legacy = Join-Path $repoRoot 'external/farina/adapt_decomp_legacy'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $reference)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $reference) | Out-Null
    Invoke-Checked git @('clone', $manifest.adapt_decomp.repository, $reference)
    Invoke-Checked git @('-C', $reference, 'checkout', '--detach', $manifest.adapt_decomp.commit)
}
$actualRevision = (& git -C $reference rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read reference checkout' }
if ($actualRevision -ne $manifest.adapt_decomp.commit) {
    throw 'Reference checkout differs from sources.json; preserve it and use a separate pinned checkout.'
}
if (-not (Test-Path -LiteralPath $legacy)) {
    Invoke-Checked git @('-C', $reference, 'worktree', 'add', '--detach', $legacy,
        $manifest.adapt_decomp.legacy_commit_for_tutorial)
}
$uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uvPath) {
    $uvPath = Join-Path $env:USERPROFILE '.local/bin/uv.exe'
    if (-not (Test-Path -LiteralPath $uvPath)) { throw 'Install uv before running setup.ps1' }
}
$benchmarkPython = Join-Path $repoRoot '.venv-benchmarks/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $benchmarkPython)) {
    Invoke-Checked $uvPath @('venv', '.venv-benchmarks', '--python', '3.12')
}
$lockName = if ($Fdsi) { 'requirements-fdsi.lock.txt' } else { 'requirements.lock.txt' }
Invoke-Checked $uvPath @('pip', 'sync', '--python', $benchmarkPython,
    (Join-Path $PSScriptRoot $lockName))
Write-Output 'Ready. Follow benchmarks/farina/README.md for the tutorial or FDSI cached reproduction.'
