# Mirrors deploy/run_trading_cycle.sh — both modes always attempt,
# overall exit is non-zero if either failed.

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Log = Join-Path $RepoRoot "logs\trading_cycle.log"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $RepoRoot

"[$(Get-Date -Format o)] trading_cycle start" | Out-File -Append $Log

& $PythonExe -m src.orchestrator --mode=paper *>> $Log
$paperExit = $LASTEXITCODE
& $PythonExe -m src.orchestrator --mode=real *>> $Log
$realExit = $LASTEXITCODE

"[$(Get-Date -Format o)] trading_cycle end (paper=$paperExit real=$realExit)" | Out-File -Append $Log
if ($paperExit -ne 0 -or $realExit -ne 0) { exit 1 }
exit 0
