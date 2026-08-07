# ============================================================================
# Runs all 4th-Aug follow-up experiments: multi-seed reruns, MPNet variants,
# Amazon V2/V3, and the two Ali-CCP V3-hybrid / significance-test helpers.
# Every run's console output is saved to a .log file under .\logs\ (next to
# this script), in addition to still printing to the terminal, so nothing
# is lost when the terminal scrolls or closes.
#
# Usage (from D:\Study, with the venv already activated, as in your screenshot):
#   .\Dissertation\run_all_experiments.ps1
#
# This will take a long time (35+ individual training runs) — it's meant
# to be started and left running, not watched live. Check the .\logs\
# folder afterwards; each file is named <script>_seed<N>.log.
# ============================================================================

$ErrorActionPreference = "Continue"
# Without this, PowerShell wraps every line a native command (python) writes
# to stderr into a red "NativeCommandError" — including harmless warnings,
# tqdm progress bars, and sentence-transformers logging, none of which mean
# the script actually failed. This makes stderr print as plain text instead,
# same as it would in a normal terminal (only available in PowerShell 7.3+;
# harmless if the variable doesn't exist on older versions).
$PSNativeCommandUseErrorActionPreference = $false

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$seeds = @(42, 123, 2026, 7, 99)

function Run-WithSeeds($scriptPath, $seeds) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($scriptPath)
    foreach ($seed in $seeds) {
        $logFile = Join-Path $logDir "${name}_seed${seed}.log"
        # Skip if this exact run already completed cleanly last time (log
        # exists and doesn't end in a Python traceback) — makes it safe/cheap
        # to rerun this whole script after a fix without redoing finished work.
        if ((Test-Path $logFile) -and -not (Select-String -Path $logFile -Pattern "Traceback \(most recent call last\)" -Quiet)) {
            Write-Host "`n=== Skipping $scriptPath --seed $seed (already completed: $logFile) ===" -ForegroundColor DarkGray
            continue
        }
        Write-Host "`n=== Running $scriptPath --seed $seed  ->  $logFile ===" -ForegroundColor Cyan
        python $scriptPath --seed $seed *>&1 | Tee-Object -FilePath $logFile
    }
}

function Run-Once($scriptPath, $extraArgs = @()) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($scriptPath)
    $logFile = Join-Path $logDir "${name}.log"
    Write-Host "`n=== Running $scriptPath $extraArgs  ->  $logFile ===" -ForegroundColor Cyan
    python $scriptPath @extraArgs *>&1 | Tee-Object -FilePath $logFile
}

# ------------------------------------------------------------------
# Step 1+2: Ali-CCP, multi-seed (baseline, V1, V1-Full, V2, and the three
# MPNet variants)
# ------------------------------------------------------------------
$aliccpScripts = @(
    "Dissertation\src\baselines\id_embedding_baseline.py",
    "Dissertation\src\models\llm_encoder_v1.py",
    "Dissertation\src\models\llm_encoder_v1_full.py",
    "Dissertation\src\models\llm_encoder_v2_aligned.py",
    "Dissertation\src\models\llm_encoder_v1_mpnet.py",
    "Dissertation\src\models\llm_encoder_v1_full_mpnet.py",
    "Dissertation\src\models\llm_encoder_v2_mpnet.py"
)
foreach ($script in $aliccpScripts) {
    Run-WithSeeds $script $seeds
}

# ------------------------------------------------------------------
# Step 3: Amazon — segmentation first (no seed), then multi-seed
# ------------------------------------------------------------------
Run-Once "Dissertation\data\preprocessing\amazon_build_difficulty_segments.py"

$amazonScripts = @(
    "Dissertation\src\baselines\amazon_id_baseline.py",
    "Dissertation\src\models\amazon_text_embedding.py",
    "Dissertation\src\models\amazon_v2_aligned.py"
)
foreach ($script in $amazonScripts) {
    Run-WithSeeds $script $seeds
}

# ------------------------------------------------------------------
# Step 4: V3-Hybrid routers (single run each, no seed — pure inference
# over the already-trained seed=42 checkpoints from steps 1-3 above)
# ------------------------------------------------------------------
Run-Once "Dissertation\src\models\llm_encoder_v3_hybrid.py"
Run-Once "Dissertation\src\models\amazon_v3_hybrid.py"

Write-Host "`n=== ALL DONE. Logs are in $logDir ===" -ForegroundColor Green
Write-Host "Next: run aggregate_multiseed_results.py / significance_tests.py per model (see their docstrings for --pattern examples)."
