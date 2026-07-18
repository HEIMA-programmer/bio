param(
    [string]$PythonExe = "",
    [string]$ScibPython = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataDir = Join-Path $RepoRoot "data"
$LogDir = Join-Path $DataDir "patient_fulltime_logs"
$StatusPath = Join-Path $DataDir "patient_fulltime_status.json"
if (-not $PythonExe) {
    $PythonExe = Join-Path $env:USERPROFILE "miniconda3\envs\scatlasvae\python.exe"
}
if (-not $ScibPython) {
    $ScibPython = Join-Path $env:USERPROFILE "miniconda3\envs\scib\python.exe"
}
foreach ($pythonPath in @($PythonExe, $ScibPython)) {
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "Python executable not found: $pythonPath. Pass the corresponding parameter explicitly."
    }
}
$StartedAt = (Get-Date).ToString("o")

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$env:NUMBA_CACHE_DIR = Join-Path $RepoRoot ".runtime_cache\numba"
$env:MPLCONFIGDIR = Join-Path $RepoRoot ".runtime_cache\matplotlib"
$env:MPLBACKEND = "Agg"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:NUMBA_NUM_THREADS = "4"

function Write-Status {
    param(
        [string]$Status,
        [string]$Message = ""
    )
    [ordered]@{
        status = $Status
        step = "transfer_patient_fulltime"
        message = $Message
        runner_pid = $PID
        started_at = $StartedAt
        updated_at = (Get-Date).ToString("o")
        expected_outputs = @(
            "phase5_transfer_results_patient_fulltime.csv",
            "phase5_transfer_cm_patient_fulltime.npz",
            "ref_model_designP_fulltime.pt",
            "..\reports\figures\fig_phase5_transfer_patient_protocol.png"
        )
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

$StdoutPath = Join-Path $LogDir "transfer_patient_fulltime.stdout.log"
$StderrPath = Join-Path $LogDir "transfer_patient_fulltime.stderr.log"

try {
    Write-Status -Status "running" -Message "Training design P with the full-time 150-epoch classifier schedule."
    Push-Location $DataDir
    try {
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PythonExe "..\scripts\phase5_annotation_transfer.py" `
            "--protocol" "fulltime" "--designs" "P" "--max-epoch" "150" `
            1> $StdoutPath 2> $StderrPath
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "Design P full-time failed with exit code $exitCode. See $StderrPath"
    }
    Push-Location $DataDir
    try {
        # Canonical figure provenance is an all-or-nothing manifest. Rebuild all
        # report figures so the changed protocol CSV cannot leave a stale manifest.
        & $ScibPython "..\scripts\figgen\build_real.py" "all"
        $figureExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($figureExitCode -ne 0) {
        throw "Canonical figure regeneration failed with exit code $figureExitCode."
    }
    Push-Location $RepoRoot
    try {
        & $ScibPython "scripts\validate_figure_manifest.py"
        $figureValidationExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($figureValidationExitCode -ne 0) {
        throw "Canonical figure manifest validation failed with exit code $figureValidationExitCode."
    }
    Write-Status -Status "complete" -Message "Design P full-time training, evaluation, and canonical figure regeneration completed successfully."
    exit 0
}
catch {
    Write-Status -Status "failed" -Message $_.Exception.Message
    exit 1
}
