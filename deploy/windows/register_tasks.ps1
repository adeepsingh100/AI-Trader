# One-time setup. Run in an elevated (Administrator) PowerShell prompt:
#   powershell -ExecutionPolicy Bypass -File deploy\windows\register_tasks.ps1
#
# Creates 3 Windows Scheduled Tasks that run this repo's trading pipeline
# 24/7 — same 3 jobs as deploy/README.md's GCP setup, same cadence.
# Runs as the current user (prompted for password below) with "run
# whether user is logged on or not" so it survives logout/lock screen,
# and auto-restarts a task that crashes.
#
# Before running this: create the venv and make sure .env exists at the
# repo root (same one used for local dev) with DATABASE_URL etc.
#   python -m venv .venv
#   .venv\Scripts\pip install -r requirements.txt

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "No venv at $PythonExe. Run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot ".env"))) {
    Write-Error "No .env at $RepoRoot\.env — the bot has no DATABASE_URL/API keys without it."
    exit 1
}

$Cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" `
    -Message "Windows password for $env:USERDOMAIN\$env:USERNAME (stored by Task Scheduler, not by this script, so tasks run even when logged out)"

function Register-Job($Name, $ScriptFile, $IntervalMinutes) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RepoRoot\deploy\windows\$ScriptFile`"" `
        -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
        -RepetitionDuration ([TimeSpan]::MaxValue)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBattery -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Minutes ([Math]::Max($IntervalMinutes * 3, 15)))
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings `
        -User $Cred.UserName -Password $Cred.GetNetworkCredential().Password -RunLevel Highest -Force | Out-Null
    Write-Host "Registered: $Name (every $IntervalMinutes min)"
}

Register-Job -Name "AI-Trader-TradingCycle" -ScriptFile "run_trading_cycle.ps1" -IntervalMinutes 10
Register-Job -Name "AI-Trader-RiskCheck"    -ScriptFile "run_risk_check.ps1"    -IntervalMinutes 1
Register-Job -Name "AI-Trader-Evolution"    -ScriptFile "run_evolution.ps1"     -IntervalMinutes 60

Write-Host ""
Write-Host "Done. If GCP's Cloud Scheduler jobs are still active, PAUSE them now" -ForegroundColor Yellow
Write-Host "(deploy/README.md) — running both at once double-processes trades," -ForegroundColor Yellow
Write-Host "there's no cross-machine guard against that." -ForegroundColor Yellow
Write-Host ""
Write-Host "Verify: Get-ScheduledTask -TaskName 'AI-Trader-*' | Get-ScheduledTaskInfo"
Write-Host "Logs land in $RepoRoot\logs\*.log"
