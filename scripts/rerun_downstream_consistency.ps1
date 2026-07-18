param(
    [string]$ScAtlasPython = "",
    [string]$ScibPython = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataDir = Join-Path $RepoRoot "data"
$LogDir = Join-Path $DataDir "downstream_pipeline_logs"
$StatusPath = Join-Path $DataDir "downstream_pipeline_status.json"
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
    param([string]$Status, [string]$Step, [string]$Message = "")
    [ordered]@{
        status = $Status
        step = $Step
        message = $Message
        runner_pid = $PID
        started_at = $StartedAt
        updated_at = (Get-Date).ToString("o")
        completed_steps = @($CompletedSteps)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-PipelineStep {
    param(
        [string]$Name,
        [string]$PythonExe,
        [string[]]$Arguments,
        [string]$WorkingDirectory = $DataDir
    )
    Write-PipelineStatus -Status "running" -Step $Name
    $stdoutPath = Join-Path $LogDir ($Name + ".stdout.log")
    $stderrPath = Join-Path $LogDir ($Name + ".stderr.log")
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
}

try {
    Invoke-PipelineStep -Name "phase3_compare_current_official" -PythonExe $ScAtlasPython -Arguments @(
        "..\scripts\phase3_train_and_compare.py", "--stage", "compare"
    )
    Invoke-PipelineStep -Name "minimal_benchmark_current_official" -PythonExe $ScibPython -Arguments @(
        "..\scripts\phase5_bench_minimal.py", "--n-jobs", "4"
    )
    Invoke-PipelineStep -Name "regenerate_downstream_figures" -PythonExe $ScibPython -Arguments @(
        "scripts\figgen\build_real.py", "umap_compare", "bench_minimal"
    ) -WorkingDirectory $RepoRoot
    Write-PipelineStatus -Status "complete" -Step "all" -Message "Downstream derived results are consistent with the corrected official embedding."
    exit 0
}
catch {
    Write-PipelineStatus -Status "failed" -Step "pipeline" -Message $_.Exception.Message
    exit 1
}
