# ============================================================================
# Runs amazon_dynamic_graph_v4_id_only.py (the ID-only ablation of V4, no
# LLM/text branch) under the same 5 seeds as everything else in this repo.
# Isolates whether the temporal-graph aggregator itself helps over the
# plain ID baseline, independent of the LLM-alignment contribution V4 and
# V2 both have.
#
# Usage (from D:\Study, with the venv already activated):
#   .\Dissertation\run_v4_id_only_experiments.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$seeds = @(42, 123, 2026, 7, 99)
$scriptPath = "Dissertation\src\models\amazon_dynamic_graph_v4_id_only.py"
$name = "amazon_dynamic_graph_v4_id_only"

foreach ($seed in $seeds) {
    $logFile = Join-Path $logDir "${name}_seed${seed}.log"
    if ((Test-Path $logFile) -and -not (Select-String -Path $logFile -Pattern "Traceback \(most recent call last\)" -Quiet)) {
        Write-Host "`n=== Skipping $scriptPath --seed $seed (already completed: $logFile) ===" -ForegroundColor DarkGray
        continue
    }
    Write-Host "`n=== Running $scriptPath --seed $seed  ->  $logFile ===" -ForegroundColor Cyan
    python $scriptPath --seed $seed *>&1 | Tee-Object -FilePath $logFile
}

Write-Host "`n=== V4-ID-ONLY ALL SEEDS DONE. Logs are in $logDir ===" -ForegroundColor Green
Write-Host "Next: python Dissertation\src\aggregate_multiseed_results.py --pattern ""D:/Study/migration_package/processed_data/amazon/v4_id_only_results/amazon_dynamic_graph_v4_id_only_metrics*.json"" --label ""Amazon V4-ID-Only"""
