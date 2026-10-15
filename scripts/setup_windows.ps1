param([switch]$Validate)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvExecutable = $uvCommand.Source
} else {
    $uvExecutable = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uvExecutable -PathType Leaf)) {
        throw 'Install uv with: winget install --id=astral-sh.uv -e; then reopen PowerShell.'
    }
}

# A Linux .venv may exist in a shared Windows/WSL checkout. Keep this one separate.
$env:UV_PROJECT_ENVIRONMENT = Join-Path $taskRoot '.venv-windows'
Push-Location -LiteralPath $taskRoot
try {
    & $uvExecutable sync --locked --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    if ($Validate) {
        & $uvExecutable run --locked pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'Simulation tests failed.' }
        & $uvExecutable run --locked python scripts/run_simulation.py --steps 200
        if ($LASTEXITCODE -ne 0) { throw 'Native simulation smoke check failed.' }
    }
    Write-Output 'Ready: .\.venv-windows\Scripts\python.exe scripts/visualize_hand.py'
} finally {
    Pop-Location
}
