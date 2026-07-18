param(
    [string]$ScAtlasPython = "",
    [string]$ScibPython = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataDir = Join-Path $RepoRoot "data"
$LogDir = Join-Path $DataDir "remaining_pipeline_logs"
$StatusPath = Join-Path $DataDir "remaining_pipeline_status.json"
$DefaultEnvRoot = Join-Path $env:USERPROFILE "miniconda3\envs"
if (-not $ScAtlasPython) { $ScAtlasPython = Join-Path $DefaultEnvRoot "scatlasvae\python.exe" }
if (-not $ScibPython) { $ScibPython = Join-Path $DefaultEnvRoot "scib\python.exe" }
foreach ($pythonPath in @($ScAtlasPython, $ScibPython)) {
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
    Invoke-PipelineStep -Name "transfer_patient_paper" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_annotation_transfer.py", "--protocol", "paper", "--designs", "P"
    )
    Invoke-PipelineStep -Name "cross_atlas_full_classifier_guarded" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_cross_atlas.py", "--max-epoch", "100", "--batch-hidden-dim", "64", "--lr", "3e-5"
    )
    Invoke-PipelineStep -Name "cross_atlas_last10_classifier_guarded" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_cross_atlas.py", "--max-epoch", "100", "--batch-hidden-dim", "64", "--lr", "3e-5", "--pred-last", "10"
    )
    Invoke-PipelineStep -Name "phase4_train" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase4_ablations.py", "--stage", "train"
    )
    Invoke-PipelineStep -Name "phase4_benchmark" -PythonExe $ScibPython -Arguments @(
        "..\scripts\phase4_ablations.py", "--stage", "benchmark", "--n-jobs", "4"
    )
    Invoke-PipelineStep -Name "scalability_complete_memory" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase5_scalability.py", "--sizes", "10000", "30000", "60000", "100000",
        "--epochs", "20", "--memory-sample-interval-ms", "50"
    )
    Invoke-PipelineStep -Name "regenerate_all_affected_figures" -PythonExe $ScibPython -Arguments @(
        "scripts\figgen\build_real.py", "bench", "umap_integration", "transfer", "invariance",
        "cross_atlas", "ablation", "scalability"
    ) -WorkingDirectory $RepoRoot
    Write-PipelineStatus -Status "complete" -Step "all" -Message "All remaining corrected reruns completed successfully."
    exit 0
}
catch {
    Write-PipelineStatus -Status "failed" -Step "pipeline" -Message $_.Exception.Message
    exit 1
}
