param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataDir = Join-Path $RepoRoot "data"
$LogDir = Join-Path $DataDir "patient_fulltime_logs"
$StatusPath = Join-Path $DataDir "patient_fulltime_status.json"
if (-not $PythonExe) {
    $PythonExe = Join-Path $env:USERPROFILE "miniconda3\envs\scatlasvae\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe. Pass -PythonExe explicitly."
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
            "ref_model_designP_fulltime.pt"
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
    Write-Status -Status "complete" -Message "Design P full-time training and evaluation completed successfully."
    exit 0
}
catch {
    Write-Status -Status "failed" -Message $_.Exception.Message
    exit 1
}
