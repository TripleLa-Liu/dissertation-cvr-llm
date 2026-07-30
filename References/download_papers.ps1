﻿# ============================================================
# Paper PDF Batch Downloader
# Run in PowerShell: Set-ExecutionPolicy Bypass -Scope Process
#                    cd to this folder, then: .\download_papers.ps1
# ============================================================

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$headers = @{ "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" }

function Download-Paper($url, $dest, $name) {
    $fullDest = Join-Path $base $dest
    if (Test-Path $fullDest) {
        Write-Host "  [SKIP] $name" -ForegroundColor Gray
        return
    }
    try {
        Write-Host "  [GET]  $name ..." -NoNewline
        Invoke-WebRequest -Uri $url -OutFile $fullDest -Headers $headers -TimeoutSec 60
        Write-Host " OK" -ForegroundColor Green
    } catch {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "        $url"
    }
}

Write-Host ""
Write-Host "=== Theme 1: CVR Prediction ===" -ForegroundColor Cyan
Download-Paper "https://arxiv.org/pdf/1703.04247" "01_CVR_Prediction\01_DeepFM_IJCAI2017.pdf" "DeepFM (IJCAI 2017)"
Download-Paper "https://arxiv.org/pdf/1706.06978" "01_CVR_Prediction\02_DIN_KDD2018.pdf" "DIN (KDD 2018)"
Download-Paper "https://arxiv.org/pdf/1809.03672" "01_CVR_Prediction\03_DIEN_AAAI2019.pdf" "DIEN (AAAI 2019)"
Download-Paper "https://arxiv.org/pdf/1804.07931" "01_CVR_Prediction\04_ESMM_SIGIR2018.pdf" "ESMM (SIGIR 2018)"
Write-Host "  [MANUAL] DFM (Chapelle, KDD 2014) - no arXiv version" -ForegroundColor Yellow
Write-Host "           https://dl.acm.org/doi/10.1145/2623330.2623634" -ForegroundColor Yellow

Write-Host ""
Write-Host "=== Theme 2: Delayed Feedback ===" -ForegroundColor Cyan
Download-Paper "https://arxiv.org/pdf/2012.03245" "02_Delayed_Feedback\06_ES-DFM_AAAI2021.pdf" "ES-DFM (AAAI 2021)"
Download-Paper "https://arxiv.org/pdf/2002.02068" "02_Delayed_Feedback\07_FSIW_WWW2020.pdf" "FSIW (WWW 2020)"
Download-Paper "https://arxiv.org/pdf/2107.05194" "02_Delayed_Feedback\08_DEFER.pdf" "DEFER"
Download-Paper "https://arxiv.org/pdf/2202.06472" "02_Delayed_Feedback\09_AsymUnbiased_2022.pdf" "Asymptotically Unbiased (2022)"
Download-Paper "https://arxiv.org/pdf/2307.12756" "02_Delayed_Feedback\10_UnbiasedLabelCorr_2023.pdf" "Unbiased Label Correction (2023)"

Write-Host ""
Write-Host "=== Theme 3: GNN RecSys ===" -ForegroundColor Cyan
Download-Paper "https://arxiv.org/pdf/2104.07368" "03_GNN_RecSys\11_DGSR_TKDE2022.pdf" "DGSR (IEEE TKDE 2022)"
Download-Paper "https://arxiv.org/pdf/2106.14226" "03_GNN_RecSys\12_SURGE_SIGIR2021.pdf" "SURGE (SIGIR 2021)"
Download-Paper "https://arxiv.org/pdf/2006.10637" "03_GNN_RecSys\13_TGN_ICML2020.pdf" "TGN (ICML Workshop 2020)"
Download-Paper "https://arxiv.org/pdf/2002.02126" "03_GNN_RecSys\14_LightGCN_SIGIR2020.pdf" "LightGCN (SIGIR 2020)"

Write-Host ""
Write-Host "=== Theme 4: LLM Recommendation ===" -ForegroundColor Cyan
Download-Paper "https://arxiv.org/pdf/2206.05941" "04_LLM_Recommendation\15_UniSRec_KDD2022.pdf" "UniSRec (KDD 2022)"
Download-Paper "https://arxiv.org/pdf/2310.15950" "04_LLM_Recommendation\16_RLMRec_WWW2024.pdf" "RLMRec (WWW 2024)"
Download-Paper "https://arxiv.org/pdf/2305.00447" "04_LLM_Recommendation\17_TALLRec_RecSys2023.pdf" "TALLRec (RecSys 2023)"
Download-Paper "https://arxiv.org/pdf/1904.06690" "04_LLM_Recommendation\18_BERT4Rec_CIKM2019.pdf" "BERT4Rec (CIKM 2019)"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Done! 17 papers auto-downloaded." -ForegroundColor Cyan
Write-Host "DFM (paper #5) needs manual download from ACM." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
