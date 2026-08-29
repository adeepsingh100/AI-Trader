# Mirrors deploy/run_evolution.sh — one sequential chain, stop on first
# failure.

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Log = Join-Path $RepoRoot "logs\evolution.log"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $RepoRoot

$steps = @(
    "src.agents.evolution_agent",
    "src.learning.adaptive_strategy_engine",
    "src.learning.drift_detection",
    "src.learning.strategy_health"
)

foreach ($step in $steps) {
    & $PythonExe -m $step *>> $Log
    if ($LASTEXITCODE -ne 0) {
        "[$(Get-Date -Format o)] evolution FAILED at $step" | Out-File -Append $Log
        exit 1
    }
}
exit 0
