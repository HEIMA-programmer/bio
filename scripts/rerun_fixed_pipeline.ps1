param(
    [string]$ScAtlasPython = "",
    [string]$ScviPython = "",
    [string]$ScibPython = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataDir = Join-Path $RepoRoot "data"
$LogDir = Join-Path $DataDir "fixed_pipeline_logs"
$StatusPath = Join-Path $DataDir "fixed_pipeline_status.json"
$DefaultEnvRoot = Join-Path $env:USERPROFILE "miniconda3\envs"
if (-not $ScAtlasPython) { $ScAtlasPython = Join-Path $DefaultEnvRoot "scatlasvae\python.exe" }
if (-not $ScviPython) { $ScviPython = Join-Path $DefaultEnvRoot "scvi\python.exe" }
if (-not $ScibPython) { $ScibPython = Join-Path $DefaultEnvRoot "scib\python.exe" }
foreach ($pythonPath in @($ScAtlasPython, $ScviPython, $ScibPython)) {
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "Python executable not found: $pythonPath. Pass the corresponding *Python parameter explicitly."
    }
}
$StartedAt = (Get-Date).ToString("o")
$CompletedSteps = New-Object System.Collections.Generic.List[string]

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$env:NUMBA_CACHE_DIR = Join-Path $RepoRoot ".runtime_cache\numba"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".runtime_cache\matplotlib"
$env:MPLBACKEND = "Agg"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:NUMBA_NUM_THREADS = "4"

function Write-PipelineStatus {
    param(
        [string]$Status,
        [string]$Step,
        [string]$Message = ""
    )
    $payload = [ordered]@{
        status = $Status
        step = $Step
        message = $Message
        runner_pid = $PID
        started_at = $StartedAt
        updated_at = (Get-Date).ToString("o")
        completed_steps = @($CompletedSteps)
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-PipelineStep {
    param(
        [string]$Name,
        [string]$PythonExe,
        [string[]]$Arguments,
        [string]$WorkingDirectory = $DataDir
    )
    Write-PipelineStatus -Status "running" -Step $Name
    $safeName = $Name -replace "[^A-Za-z0-9_.-]", "_"
    $stdoutPath = Join-Path $LogDir ($safeName + ".stdout.log")
    $stderrPath = Join-Path $LogDir ($safeName + ".stderr.log")
    Push-Location $WorkingDirectory
    try {
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PythonExe @Arguments 1> $stdoutPath 2> $stderrPath
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "Step '$Name' failed with exit code $exitCode. See $stderrPath"
    }
    $CompletedSteps.Add($Name)
    Write-PipelineStatus -Status "running" -Step $Name -Message "completed"
}

try {
    Invoke-PipelineStep -Name "transfer_paper" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_annotation_transfer.py", "--protocol", "paper", "--designs", "A", "B"
    )
    Invoke-PipelineStep -Name "transfer_fulltime" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_annotation_transfer.py", "--protocol", "fulltime", "--max-epoch", "150", "--designs", "A", "B"
    )
    Invoke-PipelineStep -Name "fair_knn_reference_encoder" -PythonExe $ScviPython -Arguments @(
        "..\scripts\phase5_fair_knn.py"
    )
    Invoke-PipelineStep -Name "cross_atlas_full_classifier" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_cross_atlas.py", "--max-epoch", "100", "--batch-hidden-dim", "64", "--lr", "3e-5"
    )
    Invoke-PipelineStep -Name "cross_atlas_last10_classifier" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_cross_atlas.py", "--max-epoch", "100", "--batch-hidden-dim", "64", "--lr", "3e-5", "--pred-last", "10"
    )
    Invoke-PipelineStep -Name "scatlasvae_invariance_probe" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_batch_invariance_probe.py", "--model", "scatlasvae"
    )
    Invoke-PipelineStep -Name "regenerate_corrected_figures" -PythonExe $ScibPython -Arguments @(
        "scripts\figgen\build_real.py", "bench", "umap_integration", "transfer", "invariance", "cross_atlas"
    ) -WorkingDirectory $RepoRoot
    Write-PipelineStatus -Status "complete" -Step "all" -Message "All corrected reruns completed successfully."
    exit 0
}
catch {
    Write-PipelineStatus -Status "failed" -Step "pipeline" -Message $_.Exception.Message
    exit 1
}
