# Reverses register_tasks.ps1 — stops and removes all 3 scheduled tasks.
# Run in an elevated PowerShell prompt.

foreach ($name in @("AI-Trader-TradingCycle", "AI-Trader-RiskCheck", "AI-Trader-Evolution")) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed: $name"
}
