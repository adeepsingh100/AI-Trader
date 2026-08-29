# Mirrors deploy/run_risk_check.sh — paper and real each run risk-only
# then diagnostics as their own chain, stopping at that chain's first
# failure; a paper failure must not skip the real chain, and vice versa.
# Runs every 1 min (see register_tasks.ps1) — orchestrator.run_risk_check
# is itself guarded by a DB mutex (risk_check_lock, migration 0015) so an
# overlapping run skips instead of double-closing a position.

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Log = Join-Path $RepoRoot "logs\risk_check.log"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $RepoRoot

$status = 0

& $PythonExe -m src.orchestrator --mode=paper --risk-only *>> $Log
if ($LASTEXITCODE -eq 0) {
    & $PythonExe -m src.monitoring.diagnostics --mode=paper *>> $Log
    if ($LASTEXITCODE -ne 0) { $status = 1 }
} else {
    $status = 1
}

& $PythonExe -m src.orchestrator --mode=real --risk-only *>> $Log
if ($LASTEXITCODE -eq 0) {
    & $PythonExe -m src.monitoring.diagnostics --mode=real *>> $Log
    if ($LASTEXITCODE -ne 0) { $status = 1 }
} else {
    $status = 1
}

exit $status
