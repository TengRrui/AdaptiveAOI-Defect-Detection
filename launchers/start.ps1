[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoDialog,
    [switch]$ValidateOnly,
    [string]$SystemRootOverride = ''
)

$ErrorActionPreference = 'Stop'
$LauncherRoot = $PSScriptRoot
$DefaultSystemRoot = Split-Path -Parent $LauncherRoot
$SystemRoot = if ($SystemRootOverride) { [System.IO.Path]::GetFullPath($SystemRootOverride) } else { $DefaultSystemRoot }
$ParentRoot = Split-Path -Parent $SystemRoot
$WorkspaceRoot = Split-Path -Parent $ParentRoot
$AssetRoot = Join-Path $LauncherRoot 'assets'
$LogRoot = Join-Path $SystemRoot 'outputs\system_runtime\logs\launcher'
$StatePath = Join-Path $LogRoot 'launcher_state.json'
$RuntimeCachePath = Join-Path $LogRoot 'runtime_cache.json'
$VideoRuntimeCachePath = Join-Path $LogRoot 'video_runtime_check_cache.json'
$WatchdogPath = Join-Path $LauncherRoot 'watchdog.ps1'
$WatchdogStopPath = Join-Path $LogRoot 'watchdog.stop'
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogPath = Join-Path $LogRoot "AdaptiveAOI_start_$Timestamp.log"
$ApiPort = if ($env:ADAPTIVEAOI_BACKEND_PORT) { [int]$env:ADAPTIVEAOI_BACKEND_PORT } else { 8000 }
$FrontendPort = if ($env:ADAPTIVEAOI_FRONTEND_PORT) { [int]$env:ADAPTIVEAOI_FRONTEND_PORT } else { 8501 }
$ApiUrl = "http://127.0.0.1:$ApiPort"
$UiUrl = "http://127.0.0.1:$FrontendPort"
$NormalizedSystemRoot = ([System.IO.Path]::GetFullPath($SystemRoot) -replace '[\\/]+$', '')
# The front-end config expands this process-local value.  Backend and UI ports
# therefore remain one launch unit even when the default 8000 port is changed.
$env:ADAPTIVEAOI_INTEGRATION_API_URL = $ApiUrl
$CreatedBackendPid = $null
$CreatedFrontendPid = $null
$WatchdogPid = $null
$LaunchStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-LauncherLog {
    param([string]$Level, [string]$Message)
    $line = "{0} [T+{1,7:N1}ms] [{2}] {3}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $LaunchStopwatch.Elapsed.TotalMilliseconds, $Level, $Message
    # A report viewer can briefly hold the active log file open on Windows.
    # Logging must never turn that transient sharing violation into a failed
    # launch after the UI/back-end processes have already been created.
    try { Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8 -ErrorAction Stop }
    catch { [System.Diagnostics.Debug]::WriteLine("AdaptiveAOI launcher log unavailable: $($_.Exception.Message)") }
}

function Normalize-ProcessPathVariable {
    # A hidden PowerShell instance may inherit both PATH and Path from the
    # desktop host.  Windows PowerShell's Start-Process rejects that duplicate
    # dictionary key, preventing Streamlit from starting.  This touches only
    # the current launcher process, never the user or machine environment.
    $pathValue = [Environment]::GetEnvironmentVariable('Path', 'Process')
    if (-not $pathValue) { $pathValue = $env:Path }
    [Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
    [Environment]::SetEnvironmentVariable('Path', $pathValue, 'Process')
}

function Show-LauncherMessage {
    param([string]$Text, [string]$Title = 'AdaptiveAOI 智能缺陷检测系统', [string]$Icon = 'Error')
    if ($NoDialog) { return }
    try {
        # The shortcut launches a hidden PowerShell process.  A modal
        # MessageBox from that process can keep the launcher mutex forever if
        # the desktop suppresses the dialog, so all dialogs have a safe bound.
        $timeout = if ($env:ADAPTIVEAOI_DIALOG_TIMEOUT_SECONDS) { [Math]::Max(1, [int]$env:ADAPTIVEAOI_DIALOG_TIMEOUT_SECONDS) } else { 12 }
        $popupType = if ($Icon -eq 'Error') { 16 } else { 64 }
        Write-LauncherLog 'INFO' "显示最长 $timeout 秒的 Windows 提示窗口：$Title"
        [void]((New-Object -ComObject WScript.Shell).Popup($Text, $timeout, $Title, $popupType))
    } catch {
        Write-LauncherLog 'WARN' "无法显示 Windows 提示框：$($_.Exception.Message)"
    }
}

function Exit-LauncherError {
    param([int]$Code, [string]$Summary, [string]$Detail)
    $message = "$Summary`r`n`r`n$Detail`r`n`r`n启动日志：$LogPath"
    Write-LauncherLog 'ERROR' $message
    if ($CreatedFrontendPid) { Stop-OwnedProcess -ProcessId $CreatedFrontendPid -Role 'frontend' }
    if ($CreatedBackendPid) { Stop-OwnedProcess -ProcessId $CreatedBackendPid -Role 'backend' }
    Show-LauncherMessage -Text $message
    exit $Code
}

function Get-DescendantProcessIds {
    param([int]$RootPid)
    $queue = New-Object System.Collections.Generic.Queue[int]
    $result = New-Object System.Collections.Generic.List[int]
    $queue.Enqueue($RootPid)
    while ($queue.Count -gt 0) {
        $parent = $queue.Dequeue()
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $parent" -ErrorAction SilentlyContinue
        foreach ($child in $children) {
            $childPid = [int]$child.ProcessId
            if (-not $result.Contains($childPid)) {
                [void]$result.Add($childPid)
                $queue.Enqueue($childPid)
            }
        }
    }
    return $result
}

function Stop-OwnedProcess {
    param([int]$ProcessId, [string]$Role)
    if (-not $ProcessId) { return }
    # The PID was created by this launcher in the current invocation.  Avoid
    # WMI command-line enumeration here: it is blocked in some user contexts
    # and previously left failed launches orphaned.
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) { return }
    if ($process.ProcessName -notmatch '^python(?:\.exe)?$') {
        Write-LauncherLog 'WARN' "拒绝停止非 Python 的本次 $Role 进程 PID=$ProcessId。"
        return
    }
    try {
        # Stop-Process can throw a NullReferenceException in restricted
        # desktop sessions even for a child created by this launcher.  Use the
        # already verified .NET handle and confirm the child has exited.
        $process.Kill()
        if (-not $process.WaitForExit(5000)) {
            throw "进程在 5 秒内未退出。"
        }
        Write-LauncherLog 'INFO' "已停止本次launchers创建的 $Role 进程 PID=$ProcessId。"
    } catch {
        Write-LauncherLog 'WARN' "停止 $Role 进程 PID=$ProcessId 失败：$($_.Exception.Message)"
    }
}

function Test-HttpStatus {
    param([string]$Url, [int]$ExpectedStatus = 200, [int]$TimeoutSeconds = 1)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return $response.StatusCode -eq $ExpectedStatus
    } catch {
        return $false
    }
}

function Get-ApiHealthProbe {
    param([int]$TimeoutSeconds = 1)
    try {
        $response = Invoke-WebRequest -Uri "$ApiUrl/api/health/ready" -UseBasicParsing -TimeoutSec $TimeoutSeconds
        $payload = $response.Content | ConvertFrom-Json
        return [pscustomobject]@{
            Reachable = $true
            HttpStatus = [int]$response.StatusCode
            Success = [bool]$payload.success
            Status = [string]$payload.status
            Ready = if ($null -eq $payload.ready) { $true } else { [bool]$payload.ready }
            Payload = $payload
        }
    } catch {
        return [pscustomobject]@{
            Reachable = $false
            HttpStatus = 0
            Success = $false
            Status = 'starting'
            Ready = $false
            Payload = $null
        }
    }
}

function Test-ApiLive {
    # A cold start has no listener yet.  Probing HTTP first waits for the full
    # request timeout before the launcher is allowed to start either service.
    # The process-local netstat probe normally finishes in milliseconds.
    $listener = Get-ListeningProcessId -Port $ApiPort
    if (-not $listener) { return $false }
    try {
        $response = Invoke-WebRequest -Uri "$ApiUrl/api/health/live" -UseBasicParsing -TimeoutSec 2
        $payload = $response.Content | ConvertFrom-Json
    } catch {
        return $false
    }
    $serviceName = [string]($payload.service)
    $reportedRoot = try {
        ([System.IO.Path]::GetFullPath([string]$payload.instance_root) -replace '[\\/]+$', '')
    } catch {
        ''
    }
    $sameInstance = [string]::Equals($reportedRoot, $NormalizedSystemRoot, [System.StringComparison]::OrdinalIgnoreCase)
    return [bool](
        $response.StatusCode -eq 200 -and
        $payload.success -and
        $payload.alive -and
        $serviceName -eq 'AdaptiveAOI' -and
        $sameInstance -and
        (Test-AdaptiveAOIBackendProcess -ProcessId $listener)
    )
}

function Test-AdaptiveAOIBackendProcess {
    param([int]$ProcessId)
    if (-not $ProcessId) { return $false }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $commandLine = [string]$process.CommandLine
    $sameProjectPath = $commandLine.IndexOf($SystemRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    return [bool]($process.Name -match '^python(?:\.exe)?$' -and $commandLine -match '(run_system_api\.py|-m\s+backend)' -and $sameProjectPath)
}

function Test-AdaptiveAOIFrontendProcess {
    param([int]$ProcessId)
    if (-not $ProcessId) { return $false }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $commandLine = [string]$process.CommandLine
    $sameProjectPath = $commandLine.IndexOf($SystemRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    return [bool]($process.Name -match '^python(?:\.exe)?$' -and $commandLine -match '-m\s+streamlit' -and $sameProjectPath)
}

function Get-ProjectMutexName {
    param([string]$Prefix)
    # A global mutex makes two extracted copies suppress each other.  The
    # normalized package path is reduced to a stable mutex-safe suffix so only
    # launches of this exact copy serialize with one another.
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($NormalizedSystemRoot.ToLowerInvariant())
        $suffix = ([System.BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').Substring(0, 16)
        return "${Prefix}_$suffix"
    } finally {
        $algorithm.Dispose()
    }
}

function Wait-ForApiLive {
    param([int]$TimeoutSeconds = 120)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-ApiLive) {
            Write-LauncherLog 'INFO' '阶段 1/6：后端进程存活检查通过；模型与运行资源继续后台预热。'
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Wait-ForHttpStatus {
    param([string]$Url, [int]$TimeoutSeconds, [string]$ServiceName)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpStatus -Url $Url) {
            Write-LauncherLog 'INFO' "$ServiceName 健康检查通过：$Url"
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Get-ListeningProcessId {
    param([int]$Port)
    # Get-NetTCPConnection can spend several seconds starting its CIM provider
    # in a cold desktop PowerShell process.  netstat is process-local and gives
    # the same listener ownership without that startup penalty.
    try {
        $netstat = Join-Path $env:SystemRoot 'System32\netstat.exe'
        foreach ($line in (& $netstat -ano -p tcp 2>$null)) {
            if ($line -match ("^\s*TCP\s+\S+:" + [regex]::Escape([string]$Port) + "\s+\S+\s+LISTENING\s+(\d+)\s*$")) {
                return [int]$Matches[1]
            }
        }
    } catch {
        Write-LauncherLog 'WARN' "快速端口检查失败，将使用系统网络命令：$($_.Exception.Message)"
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($listener) { return [int]$listener.OwningProcess }
    }
    return $null
}

function Get-VideoRuntimeCheck {
    param([string]$PythonPath, [string]$CheckPath)
    $pythonStamp = (Get-Item -LiteralPath $PythonPath).LastWriteTimeUtc.Ticks
    $checkStamp = (Get-Item -LiteralPath $CheckPath).LastWriteTimeUtc.Ticks
    if (Test-Path -LiteralPath $VideoRuntimeCachePath) {
        try {
            $cached = Get-Content -LiteralPath $VideoRuntimeCachePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if (
                [string]::Equals([string]$cached.python_path, $PythonPath, [System.StringComparison]::OrdinalIgnoreCase) -and
                [int64]$cached.python_stamp -eq $pythonStamp -and
                [int64]$cached.check_stamp -eq $checkStamp
            ) {
                return [pscustomobject]@{ Success = [bool]$cached.success; Output = [string]$cached.output; CacheHit = $true }
            }
        } catch {
            Write-LauncherLog 'WARN' "视频运行时缓存不可读，将执行一次真实自检：$($_.Exception.Message)"
        }
    }
    $output = (& $PythonPath $CheckPath 2>&1) -join ' '
    $success = $LASTEXITCODE -eq 0
    @{
        python_path = $PythonPath
        python_stamp = $pythonStamp
        check_stamp = $checkStamp
        success = $success
        output = $output
        checked_at = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $VideoRuntimeCachePath -Encoding UTF8
    return [pscustomobject]@{ Success = $success; Output = $output; CacheHit = $false }
}

function Wait-ForPortReleased {
    param([int]$Port, [int]$TimeoutSeconds = 8)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-ListeningProcessId -Port $Port)) { return $true }
        Start-Sleep -Milliseconds 200
    }
    return $false
}

function Get-PreferredPython {
    $candidates = New-Object System.Collections.Generic.List[string]
    $cachedPython = $null
    if (Test-Path -LiteralPath $RuntimeCachePath) {
        try {
            $cached = Get-Content -LiteralPath $RuntimeCachePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($cached.python_path) { $cachedPython = [string]$cached.python_path }
        } catch {
            Write-LauncherLog 'WARN' "运行时缓存不可读，将自动重新发现解释器：$($_.Exception.Message)"
        }
    }

    # A known-good interpreter should not pay the cost of process enumeration,
    # Conda probing, or filesystem-drive scanning on every launch.
    if ($cachedPython -and (Test-Path -LiteralPath $cachedPython)) {
        try {
            $probe = & $cachedPython -c "import importlib.util; print('adaptive_aoi_runtime_ok' if all(importlib.util.find_spec(n) for n in ('fastapi','streamlit','torch')) else 'adaptive_aoi_runtime_missing')" 2>&1
            if ($LASTEXITCODE -eq 0 -and ($probe -join "`n") -match 'adaptive_aoi_runtime_ok') {
                return (Resolve-Path -LiteralPath $cachedPython).Path
            }
            Write-LauncherLog 'WARN' "缓存的 Python 运行时不可用，将自动重新发现：$cachedPython"
        } catch {
            Write-LauncherLog 'WARN' "缓存的 Python 运行时探测失败，将自动重新发现：$cachedPython；$($_.Exception.Message)"
        }
    }

    if ($env:ADAPTIVEAOI_PYTHON) { [void]$candidates.Add($env:ADAPTIVEAOI_PYTHON) }
    if ($env:CONDA_PREFIX) { [void]$candidates.Add((Join-Path $env:CONDA_PREFIX 'python.exe')) }

    # Prefer an already-running current system interpreter without invoking Conda.
    $runningPython = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -ieq 'python.exe' -and [string]$_.CommandLine -like "*$SystemRoot*"
    } | Select-Object -First 1
    if ($runningPython -and $runningPython.ExecutablePath) { [void]$candidates.Add($runningPython.ExecutablePath) }

    $condaCommands = @($env:CONDA_EXE)
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($condaCommand) { $condaCommands += $condaCommand.Source }
    foreach ($condaExe in ($condaCommands | Where-Object { $_ } | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $condaExe)) { continue }
        $scriptsRoot = Split-Path -Parent $condaExe
        $condaRoot = Split-Path -Parent $scriptsRoot
        [void]$candidates.Add((Join-Path $condaRoot 'envs\adaptive_aoi\python.exe'))
    }

    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        foreach ($relativePath in @('Anaconda\anaconda\envs\adaptive_aoi\python.exe', 'anaconda3\envs\adaptive_aoi\python.exe', 'miniconda3\envs\adaptive_aoi\python.exe')) {
            [void]$candidates.Add((Join-Path $drive.Root $relativePath))
        }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pythonCommand) { [void]$candidates.Add($pythonCommand.Source) }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            # Do not import torch here.  Importing the CUDA runtime made every
            # cold launch wait tens of seconds before either service was started.
            $probe = & $candidate -c "import importlib.util; print('adaptive_aoi_runtime_ok' if all(importlib.util.find_spec(n) for n in ('fastapi','streamlit','torch')) else 'adaptive_aoi_runtime_missing')" 2>&1
            if ($LASTEXITCODE -eq 0 -and ($probe -join "`n") -match 'adaptive_aoi_runtime_ok') {
                [pscustomobject]@{ python_path = (Resolve-Path -LiteralPath $candidate).Path; validated_at = (Get-Date).ToString('o'); probe = 'importlib.util.find_spec' } |
                    ConvertTo-Json | Set-Content -LiteralPath $RuntimeCachePath -Encoding UTF8
                return (Resolve-Path -LiteralPath $candidate).Path
            }
            Write-LauncherLog 'WARN' "Python 候选缺少运行依赖：$candidate"
        } catch {
            Write-LauncherLog 'WARN' "Python 候选不可用：$candidate；$($_.Exception.Message)"
        }
    }
    return $null
}

function Write-RuntimeProfileSummary {
    $probe = Get-ApiHealthProbe -TimeoutSeconds 5
    if (-not ($probe.Reachable -and $probe.Success -and $probe.Payload)) {
        Write-LauncherLog 'WARN' '暂时无法读取统一运行 Profile 摘要。'
        return
    }
    $profile = $probe.Payload.current_profile
    if (-not $profile) {
        Write-LauncherLog 'INFO' (
            "当前 Profile 正在后台解析；就绪状态={0}；阶段={1}；进度={2}%" -f `
            $probe.Payload.status, $probe.Payload.phase, $probe.Payload.progress
        )
        return
    }
    $configBinding = if ($profile.config_profile_id) {
        "$($profile.config_profile_id):v$($profile.config_profile_version)"
    } else {
        '未绑定'
    }
    # Windows PowerShell 5.1 may decode UTF-8 JSON text with the legacy ANSI
    # code page.  Keep the launcher log readable by mapping the stable ASCII
    # phase identifier locally instead of echoing the decoded stage label.
    $stageName = switch ([string]$probe.Payload.phase) {
        'api_ready' { 'API 已就绪' }
        'loading_settings' { '加载服务配置' }
        'resolving_runtime' { '解析正式运行方案' }
        'backend' { '启动后端' }
        'configuration' { '加载配置' }
        'profile' { '加载 Profile' }
        'loading_profile' { '加载 Profile' }
        'loading_runtime' { '加载模型与 Memory Bank' }
        'loading_model' { '加载模型' }
        'loading_memory_bank' { '加载 Memory Bank' }
        'building_index' { '构建近邻检索索引' }
        'warming_up' { '完成检测资源预热' }
        'initializing_index' { '初始化近邻检索索引' }
        'ready' { '预热完成' }
        'failed' { '预热失败' }
        default { '后台初始化' }
    }
    Write-LauncherLog 'INFO' (
        "当前 Profile：名称={0}；版本={1}；Memory Bank={2}；Config={3}；就绪状态={4}；阶段={5}；进度={6}%" -f `
        $profile.name, $profile.version, $profile.memory_bank_id, $configBinding, $probe.Payload.status, $stageName, $probe.Payload.progress
    )
}

function Resolve-DataRoot {
    if ($env:ADAPTIVEAOI_DATA_ROOT -and (Test-Path -LiteralPath $env:ADAPTIVEAOI_DATA_ROOT)) {
        return (Resolve-Path -LiteralPath $env:ADAPTIVEAOI_DATA_ROOT).Path
    }
    return $null
}

function Save-LauncherState {
    $backendPid = Get-ListeningProcessId -Port $ApiPort
    $frontendPid = Get-ListeningProcessId -Port $FrontendPort
    if (-not $backendPid -and $CreatedBackendPid) { $backendPid = $CreatedBackendPid }
    if (-not $frontendPid -and $CreatedFrontendPid) { $frontendPid = $CreatedFrontendPid }
    $backendProcess = if ($backendPid) { Get-Process -Id $backendPid -ErrorAction SilentlyContinue } else { $null }
    $frontendProcess = if ($frontendPid) { Get-Process -Id $frontendPid -ErrorAction SilentlyContinue } else { $null }
    $watchdogProcess = if ($WatchdogPid) { Get-Process -Id $WatchdogPid -ErrorAction SilentlyContinue } else { $null }
    $state = [pscustomobject]@{
        created_at = (Get-Date).ToString('o')
        launcher_version = '1.0.0'
        project_root = $SystemRoot
        python_path = $python
        api_url = $ApiUrl
        ui_url = $UiUrl
        backend = [pscustomobject]@{
            started_by_launcher = [bool]$backendPid
            pid = $backendPid
            process_started_at = if ($backendProcess) { $backendProcess.StartTime.ToUniversalTime().ToString('o') } else { $null }
        }
        frontend = [pscustomobject]@{
            started_by_launcher = [bool]$frontendPid
            pid = $frontendPid
            process_started_at = if ($frontendProcess) { $frontendProcess.StartTime.ToUniversalTime().ToString('o') } else { $null }
        }
        watchdog = [pscustomobject]@{
            started_by_launcher = [bool]$watchdogProcess
            pid = if ($watchdogProcess) { $WatchdogPid } else { $null }
            process_started_at = if ($watchdogProcess) { $watchdogProcess.StartTime.ToUniversalTime().ToString('o') } else { $null }
        }
    }
    $temporary = "$StatePath.launcher.tmp"
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Get-ExistingWatchdogProcessId {
    # launcher_state.json can legitimately be rewritten by a second desktop
    # launch while the original watchdog is still alive.  Discover the real
    # process from its exact script path instead of treating a missing/stale
    # state-file PID as proof that no watchdog exists.
    try {
        $processes = Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction Stop
        foreach ($process in $processes) {
            $commandLine = [string]$process.CommandLine
            $sameWatchdog = $commandLine.IndexOf($WatchdogPath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
            $sameProject = $commandLine.IndexOf($SystemRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
            if ($sameWatchdog -and $sameProject) {
                return [int]$process.ProcessId
            }
        }
    } catch {
        Write-LauncherLog 'WARN' "无法枚举现有守护进程，将继续执行受控启动：$($_.Exception.Message)"
    }
    return $null
}

function Start-BackendWatchdog {
    if (Test-Path -LiteralPath $WatchdogStopPath) {
        Remove-Item -LiteralPath $WatchdogStopPath -Force
    }
    if (Test-Path -LiteralPath $StatePath) {
        try {
            $previous = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
            $previousPid = [int]($previous.watchdog.pid)
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $previousPid" -ErrorAction SilentlyContinue
            if ($process -and $process.Name -match '^powershell(?:\.exe)?$' -and ([string]$process.CommandLine).IndexOf($WatchdogPath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $script:WatchdogPid = $previousPid
                Write-LauncherLog 'INFO' "后端守护进程已存在 PID=$previousPid，未重复启动。"
                return
            }
        } catch {
            Write-LauncherLog 'WARN' "旧守护状态不可用，将创建新守护进程：$($_.Exception.Message)"
        }
    }
    $discoveredPid = Get-ExistingWatchdogProcessId
    if ($discoveredPid) {
        $script:WatchdogPid = $discoveredPid
        Write-LauncherLog 'INFO' "已发现正在运行的后端守护进程 PID=$discoveredPid，未重复启动。"
        return
    }
    $watchdogStdout = Join-Path $LogRoot "watchdog_process_$Timestamp.stdout.log"
    $watchdogStderr = Join-Path $LogRoot "watchdog_process_$Timestamp.stderr.log"
    $watchdogArguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
        '-File', ('"{0}"' -f $WatchdogPath),
        '-SystemRoot', ('"{0}"' -f $SystemRoot),
        '-PythonPath', ('"{0}"' -f $python),
        '-ApiPort', "$ApiPort"
    )
    $watchdog = Start-Process -FilePath 'powershell.exe' -ArgumentList $watchdogArguments -WorkingDirectory $SystemRoot -WindowStyle Hidden -RedirectStandardOutput $watchdogStdout -RedirectStandardError $watchdogStderr -PassThru
    $script:WatchdogPid = $watchdog.Id
    Start-Sleep -Milliseconds 500
    $watchdog.Refresh()
    if ($watchdog.HasExited) {
        # The watchdog owns a named mutex.  A concurrently existing instance
        # makes the new process exit 0 by design; that is an idempotent launch,
        # not a startup failure.  Re-discover the owner and publish its PID so
        # later launches can reuse the state without another transient process.
        $existingPid = if ($watchdog.ExitCode -eq 0) { Get-ExistingWatchdogProcessId } else { $null }
        if ($existingPid) {
            $script:WatchdogPid = $existingPid
            Write-LauncherLog 'INFO' "守护进程互斥锁已由 PID=$existingPid 持有；本次重复启动正常结束。"
            return
        }
        $stderrDetail = if (Test-Path -LiteralPath $watchdogStderr) { Get-Content -LiteralPath $watchdogStderr -Raw -ErrorAction SilentlyContinue } else { "" }
        $stdoutDetail = if (Test-Path -LiteralPath $watchdogStdout) { Get-Content -LiteralPath $watchdogStdout -Raw -ErrorAction SilentlyContinue } else { "" }
        throw "后端守护进程启动后立即退出，退出码=$($watchdog.ExitCode)。$stderrDetail$stdoutDetail"
    }
    Write-LauncherLog 'INFO' "后端守护进程已启动 PID=$WatchdogPid。"
}

Write-LauncherLog 'INFO' "T1 launchers开始执行。系统目录=$SystemRoot；工作目录=$PWD"
Normalize-ProcessPathVariable

if (-not (Test-Path -LiteralPath $SystemRoot)) {
    Exit-LauncherError 10 '算法系统目录不存在。' "未找到：$SystemRoot"
}

$FrontendEntry = Join-Path $SystemRoot 'app\系统总览.py'
$SoftwareConfig = Join-Path $SystemRoot 'configs\software_config.yaml'
$BackendEntry = Join-Path $SystemRoot 'backend\__main__.py'
$IntegrationConfig = Join-Path $SystemRoot 'configs\system_integration.yaml'
$VideoRuntimeCheck = Join-Path $SystemRoot 'scripts\check_video_runtime.py'
foreach ($requiredPath in @($FrontendEntry, $SoftwareConfig, $BackendEntry, $IntegrationConfig, $VideoRuntimeCheck, $WatchdogPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        Exit-LauncherError 11 '系统启动配置不完整。' "缺少必要文件：$requiredPath"
    }
}

$dataRoot = Resolve-DataRoot
if ($dataRoot) {
    $env:ADAPTIVEAOI_DATA_ROOT = $dataRoot
    Write-LauncherLog 'INFO' "本次启动发现可用的目标域样本根：$dataRoot"
} else {
    # Frozen inference only needs the packaged profile, memory bank, model and
    # threshold.  Raw training/adaptation data is selected explicitly when an
    # operator enters the target-domain workflow, so it must not be a launcher
    # dependency or a machine-specific path requirement.
    [Environment]::SetEnvironmentVariable('ADAPTIVEAOI_DATA_ROOT', $null, 'Process')
    Write-LauncherLog 'INFO' '未配置目标域原始样本根；冻结推理可独立启动，目标域适配时由用户选择样本目录。'
}

$python = Get-PreferredPython
if (-not $python) {
    Exit-LauncherError 12 '未找到可用的 AdaptiveAOI Python 运行环境。' '请激活包含 fastapi、streamlit、torch 的 adaptive_aoi 环境，或为当前用户会话设置 ADAPTIVEAOI_PYTHON。'
}
$pythonVersion = (& $python --version 2>&1) -join ' '
Write-LauncherLog 'INFO' "使用 Python：$python；$pythonVersion"
$videoRuntime = Get-VideoRuntimeCheck -PythonPath $python -CheckPath $VideoRuntimeCheck
$videoRuntimeSource = if ($videoRuntime.CacheHit) { '缓存命中' } else { '真实校验' }
if ($videoRuntime.Success) {
    Write-LauncherLog 'INFO' "视频运行时自检通过（$videoRuntimeSource）：$($videoRuntime.Output)"
} else {
    Write-LauncherLog 'WARN' "视频运行时自检未通过（$videoRuntimeSource）；图像检测仍可启动，视频检测需先修复编解码环境：$($videoRuntime.Output)"
}

if ($ValidateOnly) {
    Write-LauncherLog 'INFO' 'launchers验证完成，未启动服务。'
    exit 0
}

$mutex = New-Object System.Threading.Mutex($false, (Get-ProjectMutexName -Prefix 'AdaptiveAOI_System_Launcher_Mutex'))
$hasMutex = $false
try {
    try { $hasMutex = $mutex.WaitOne(0) } catch { $hasMutex = $false }
    if (-not $hasMutex) {
        Write-LauncherLog 'INFO' '检测到另一个启动请求正在执行。'
        if (Test-HttpStatus -Url $UiUrl) {
            if (-not $NoBrowser) {
                try { Start-Process $UiUrl; Write-LauncherLog 'INFO' '已请求默认浏览器打开已运行的系统页面。' }
                catch { Write-LauncherLog 'WARN' ("无法自动打开默认浏览器；服务继续运行。原因：{0}" -f $_.Exception.Message) }
            }
            exit 0
        }
        Show-LauncherMessage -Text "系统正在启动，请稍候后再试。`r`n启动日志：$LogPath" -Icon Information
        exit 0
    }

    # Liveness and readiness are intentionally separate.  The UI can start as
    # soon as its own HTTP server is available.  Backend and Streamlit are
    # launched in parallel so Windows cold imports cannot leave the operator
    # with no visible page; model readiness never gates the browser.
    $backendAlreadyHealthy = Test-ApiLive
    $backendNeedsLiveWait = -not $backendAlreadyHealthy
    $backendErrorLog = $null
    if ($backendAlreadyHealthy) {
        Write-LauncherLog 'INFO' '检测到可用的 AdaptiveAOI 后端，未重复启动。'
    } else {
        $existingBackendPid = Get-ListeningProcessId -Port $ApiPort
        if ($existingBackendPid) {
            if (-not (Test-AdaptiveAOIBackendProcess -ProcessId $existingBackendPid)) {
                Exit-LauncherError 14 '后端端口被其他程序占用。' "端口 $ApiPort 已被其他程序占用；未结束该进程，请关闭占用程序后重试。"
            }
            Write-LauncherLog 'INFO' "发现本系统已有后端进程 PID=$existingBackendPid，等待其健康检查通过。"
        } else {
            $backendOut = Join-Path $LogRoot "backend_$Timestamp.stdout.log"
            $backendErrorLog = Join-Path $LogRoot "backend_$Timestamp.stderr.log"
            $backend = Start-Process -FilePath $python -ArgumentList @('-m', 'backend', '--config', ('"{0}"' -f $IntegrationConfig), '--host', '127.0.0.1', '--port', "$ApiPort") -WorkingDirectory $SystemRoot -WindowStyle Hidden -RedirectStandardOutput $backendOut -RedirectStandardError $backendErrorLog -PassThru
            $CreatedBackendPid = $backend.Id
            Write-LauncherLog 'INFO' "T2 后端进程已启动 PID=$CreatedBackendPid；后台继续初始化。"
        }
    }

    # Avoid paying the one-second HTTP timeout when no process is listening
    # yet.  The port probe is enough to decide that a fresh Streamlit process
    # must be created; only an existing listener needs the HTTP health check.
    $frontendAlreadyHealthy = $false
    $frontendListener = Get-ListeningProcessId -Port $FrontendPort
    if ($frontendListener) {
        if (-not (Test-AdaptiveAOIFrontendProcess -ProcessId $frontendListener)) {
            Exit-LauncherError 14 '前端端口被其他程序占用。' "端口 $FrontendPort 属于另一程序或另一份 AdaptiveAOI；未复用、未结束该进程，请更换端口或关闭占用程序后重试。"
        }
        $frontendAlreadyHealthy = Test-HttpStatus -Url $UiUrl
    }
    if (-not $frontendAlreadyHealthy) {
        $frontendReady = $false
        $frontendErrorLogs = New-Object System.Collections.Generic.List[string]
        foreach ($attempt in 1..2) {
            $existingFrontendPid = Get-ListeningProcessId -Port $FrontendPort
            if ($existingFrontendPid) {
                Exit-LauncherError 14 '前端端口被其他程序占用。' "端口 $FrontendPort 已被其他程序占用；未结束该进程，请关闭占用程序后重试。"
            }
            $frontendOut = Join-Path $LogRoot "frontend_${Timestamp}_attempt${attempt}.stdout.log"
            $frontendErr = Join-Path $LogRoot "frontend_${Timestamp}_attempt${attempt}.stderr.log"
            [void]$frontendErrorLogs.Add($frontendErr)
            # This is a formal desktop deployment, not a development session.
            # Disable Streamlit's recursive source watcher so the first page
            # does not wait for a project-wide file scan before the shell is
            # served.  Runtime/resource integrity checks remain owned by the
            # backend and are not skipped by this UI-only setting.
            $frontend = Start-Process -FilePath $python -ArgumentList @('-m', 'streamlit', 'run', ('"{0}"' -f $FrontendEntry), '--server.port', "$FrontendPort", '--server.address', '127.0.0.1', '--server.headless', 'true', '--browser.gatherUsageStats', 'false', '--server.fileWatcherType', 'none') -WorkingDirectory $SystemRoot -WindowStyle Hidden -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru
            $CreatedFrontendPid = $frontend.Id
            Write-LauncherLog 'INFO' "T3 前端进程已启动 PID=$CreatedFrontendPid；尝试 $attempt/2。"
            if (Wait-ForHttpStatus -Url $UiUrl -TimeoutSeconds 60 -ServiceName 'AdaptiveAOI 前端') {
                $frontendReady = $true
                break
            }
            Write-LauncherLog 'WARN' "前端尝试 $attempt/2 未通过 HTTP 健康检查；将仅停止本次创建的进程后重试。日志：$frontendErr"
            Stop-OwnedProcess -ProcessId $CreatedFrontendPid -Role 'frontend'
            if (-not (Wait-ForPortReleased -Port $FrontendPort)) {
                Exit-LauncherError 14 '前端端口未能释放。' "端口 $FrontendPort 在本次前端失败后仍被占用；请查看日志：$frontendErr"
            }
            $CreatedFrontendPid = $null
        }
        if (-not $frontendReady) {
            Exit-LauncherError 21 'AdaptiveAOI 前端未能打开。' ("已执行两次受控启动尝试；请查看前端日志：" + ($frontendErrorLogs -join '；'))
        }
    } else {
        Write-LauncherLog 'INFO' '系统已经在运行，未重复启动 Streamlit。'
    }

    Write-LauncherLog 'INFO' 'T4 前端页面健康检查通过；浏览器不等待后端 Runtime Ready。'
    # Publish owned PIDs before opening the browser so the close entry remains
    # correct even if the operator immediately requests shutdown.
    Save-LauncherState
    if (-not $NoBrowser) {
        try {
            Start-Process $UiUrl
            Write-LauncherLog 'INFO' 'T5 已请求默认浏览器打开系统页面。'
        } catch {
            Write-LauncherLog 'WARN' ("无法自动打开默认浏览器；服务继续运行。原因：{0}" -f $_.Exception.Message)
        }
    }

    # Continue the hidden launcher validation after the browser is visible.
    # A backend failure is still diagnosed, but it no longer delays the first
    # interactive Streamlit shell.
    if ($backendNeedsLiveWait) {
        $backendLiveTimeout = if ($env:ADAPTIVEAOI_BACKEND_READY_TIMEOUT_SECONDS) {
            [Math]::Max(30, [int]$env:ADAPTIVEAOI_BACKEND_READY_TIMEOUT_SECONDS)
        } else {
            120
        }
        if (-not (Wait-ForApiLive -TimeoutSeconds $backendLiveTimeout)) {
            $detail = if ($backendErrorLog) { "算法服务启动失败，请查看后端日志：$backendErrorLog" } else { '算法服务进程未通过存活检查，请查看启动日志并重试。' }
            Exit-LauncherError 20 '算法服务启动失败。' $detail
        }
    }

    # The launcher never selects resources itself.  It reports the exact
    # binding resolved by RuntimeProfileService through the health endpoint.
    Write-RuntimeProfileSummary
    Start-BackendWatchdog

    Save-LauncherState
    # Do not issue a second, potentially blocking readiness request while CUDA
    # is initializing.  The first snapshot above records the Profile and the UI
    # continues to read readiness independently.
    Write-LauncherLog 'INFO' '启动流程完成；模型与运行资源在后台继续预热，存活状态来自 /api/health/live，就绪与阶段来自 /api/health/ready。'
    exit 0
} catch {
    # Keep enough diagnostic context in the per-run log to make launcher
    # failures repairable without exposing implementation details in the UI.
    Write-LauncherLog 'ERROR' ("异常类型：{0}" -f $_.Exception.GetType().FullName)
    Write-LauncherLog 'ERROR' ("异常位置：{0}" -f ($_.InvocationInfo.PositionMessage -replace "`r?`n", ' | '))
    Exit-LauncherError 22 'launchers发生未预期错误。' $_.Exception.Message
} finally {
    if ($hasMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
