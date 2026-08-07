# ============================================================================
# Runs amazon_dynamic_graph_v4.py under 5 seeds (same set as everything else
# in this repo: 42, 123, 2026, 7, 99) so V4 can go through the same
# aggregate_multiseed_results.py / significance_tests.py pipeline as
# Baseline/V1/V2 on Amazon. Seed 42 already ran once (2026-08-07) —
# this script skips it automatically via the same log-based skip check
# run_all_experiments.ps1 uses, so it's safe to just run this as-is.
#
# Usage (from D:\Study, with the venv already activated):
#   .\Dissertation\run_v4_experiments.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false   # see run_all_experiments.ps1 for why

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$seeds = @(42, 123, 2026, 7, 99)
$scriptPath = "Dissertation\src\models\amazon_dynamic_graph_v4.py"
$name = "amazon_dynamic_graph_v4"

foreach ($seed in $seeds) {
    $logFile = Join-Path $logDir "${name}_seed${seed}.log"
    if ((Test-Path $logFile) -and -not (Select-String -Path $logFile -Pattern "Traceback \(most recent call last\)" -Quiet)) {
        Write-Host "`n=== Skipping $scriptPath --seed $seed (already completed: $logFile) ===" -ForegroundColor DarkGray
        continue
    }
    Write-Host "`n=== Running $scriptPath --seed $seed  ->  $logFile ===" -ForegroundColor Cyan
    python $scriptPath --seed $seed *>&1 | Tee-Object -FilePath $logFile
}

Write-Host "`n=== V4 ALL SEEDS DONE. Logs are in $logDir ===" -ForegroundColor Green
Write-Host "Next: python Dissertation\src\aggregate_multiseed_results.py --pattern ""D:/Study/migration_package/processed_data/amazon/v4_dynamic_graph_results/amazon_dynamic_graph_v4_metrics*.json"" --label ""Amazon V4 (Dynamic Graph)"""
