[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SystemRoot,
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [int]$ApiPort = 8000
)

$ErrorActionPreference = 'Continue'
$NormalizedSystemRoot = ([System.IO.Path]::GetFullPath($SystemRoot) -replace '[\\/]+$', '')
$env:ADAPTIVEAOI_INTEGRATION_API_URL = "http://127.0.0.1:$ApiPort"
$LogRoot = Join-Path $SystemRoot 'outputs\system_runtime\logs\launcher'
$StatePath = Join-Path $LogRoot 'launcher_state.json'
$StopSignalPath = Join-Path $LogRoot 'watchdog.stop'
$LogPath = Join-Path $LogRoot ("AdaptiveAOI_watchdog_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-WatchdogLog {
    param([string]$Level, [string]$Message)
    try {
        Add-Content -LiteralPath $LogPath -Value ("{0} [{1}] {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message) -Encoding UTF8
    } catch {
        [System.Diagnostics.Debug]::WriteLine("AdaptiveAOI watchdog log unavailable: $($_.Exception.Message)")
    }
}

function Get-ListeningProcessId {
    $listener = Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) { return [int]$listener.OwningProcess }
    return $null
}

function Test-ProjectBackend {
    param([int]$ProcessId)
    if (-not $ProcessId) { return $false }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $commandLine = [string]$process.CommandLine
    $validName = $process.Name -match '^python(?:\.exe)?$'
    $validEntry = $commandLine -match '(run_system_api\.py|-m\s+backend)'
    $validRoot = $commandLine.IndexOf($SystemRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    if ($validName -and $validEntry -and $validRoot) { return $true }
    return $false
}

function Test-BackendLive {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/api/health/live" -UseBasicParsing -TimeoutSec 2
        $payload = $response.Content | ConvertFrom-Json
        $reportedRoot = try {
            ([System.IO.Path]::GetFullPath([string]$payload.instance_root) -replace '[\\/]+$', '')
        } catch {
            ''
        }
        $sameInstance = [string]::Equals($reportedRoot, $NormalizedSystemRoot, [System.StringComparison]::OrdinalIgnoreCase)
        return [bool]($response.StatusCode -eq 200 -and $payload.success -and $payload.alive -and $sameInstance)
    } catch {
        return $false
    }
}

function Get-ProjectMutexName {
    param([string]$Prefix)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($NormalizedSystemRoot.ToLowerInvariant())
        $suffix = ([System.BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').Substring(0, 16)
        return "${Prefix}_$suffix"
    } finally {
        $algorithm.Dispose()
    }
}

function Update-BackendState {
    param([int]$ProcessId)
    try {
        if (Test-Path -LiteralPath $StatePath) {
            $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        } else {
            $state = [pscustomobject]@{}
        }
        $state | Add-Member -Force NoteProperty backend ([pscustomobject]@{
            started_by_launcher = $true
            pid = $ProcessId
            process_started_at = (Get-Process -Id $ProcessId -ErrorAction Stop).StartTime.ToUniversalTime().ToString('o')
            restarted_by_watchdog = $true
        })
        $temporary = "$StatePath.watchdog.tmp"
        $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding UTF8
        Move-Item -LiteralPath $temporary -Destination $StatePath -Force
    } catch {
        Write-WatchdogLog 'WARN' "Could not update backend state: $($_.Exception.Message)"
    }
}

$mutex = New-Object System.Threading.Mutex($false, (Get-ProjectMutexName -Prefix 'AdaptiveAOI_System_Backend_Watchdog_Mutex'))
$hasMutex = $false
try {
    try { $hasMutex = $mutex.WaitOne(0) } catch { $hasMutex = $false }
    if (-not $hasMutex) {
        Write-WatchdogLog 'INFO' 'Another backend watchdog is already active; this instance exits.'
        exit 0
    }
    if (Test-Path -LiteralPath $StopSignalPath) {
        Remove-Item -LiteralPath $StopSignalPath -Force
    }
    Write-WatchdogLog 'INFO' "Backend watchdog started; project=$SystemRoot; port=$ApiPort."
    while (-not (Test-Path -LiteralPath $StopSignalPath)) {
        if (Test-BackendLive) {
            Start-Sleep -Seconds 2
            continue
        }
        $listener = Get-ListeningProcessId
        if ($listener) {
            if (-not (Test-ProjectBackend -ProcessId $listener)) {
                Write-WatchdogLog 'ERROR' "Port $ApiPort is owned by a non-project process PID=$listener; restart skipped."
                Start-Sleep -Seconds 5
                continue
            }
            Start-Sleep -Seconds 2
            continue
        }
        if (Test-Path -LiteralPath $StopSignalPath) { break }
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $stdout = Join-Path $LogRoot "backend_watchdog_$($timestamp).stdout.log"
        $stderr = Join-Path $LogRoot "backend_watchdog_$($timestamp).stderr.log"
        try {
            $integrationConfig = Join-Path $SystemRoot 'configs\system_integration.yaml'
            $backend = Start-Process -FilePath $PythonPath -ArgumentList @('-m', 'backend', '--config', ('"{0}"' -f $integrationConfig), '--host', '127.0.0.1', '--port', "$ApiPort") -WorkingDirectory $SystemRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
            Write-WatchdogLog 'WARN' "Backend exit detected; recovery process started PID=$($backend.Id)."
            $deadline = (Get-Date).AddSeconds(180)
            while ((Get-Date) -lt $deadline -and -not (Test-Path -LiteralPath $StopSignalPath)) {
                if (Test-BackendLive) {
                    Update-BackendState -ProcessId $backend.Id
                    Write-WatchdogLog 'INFO' "Backend recovery completed PID=$($backend.Id)."
                    break
                }
                if ($backend.HasExited) {
                    Write-WatchdogLog 'ERROR' "Recovery process exited early with code=$($backend.ExitCode); stderr=$stderr"
                    break
                }
                Start-Sleep -Seconds 1
            }
        } catch {
            Write-WatchdogLog 'ERROR' "Backend recovery start failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 5
        }
    }
    Write-WatchdogLog 'INFO' 'Formal stop signal received; backend watchdog exits.'
} finally {
    if ($hasMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
