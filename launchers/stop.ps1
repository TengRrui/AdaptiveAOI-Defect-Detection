[CmdletBinding()]
param([switch]$NoDialog)

$ErrorActionPreference = 'Stop'
$LauncherRoot = $PSScriptRoot
$SystemRoot = Split-Path -Parent $LauncherRoot
$LogRoot = Join-Path $SystemRoot 'outputs\system_runtime\logs\launcher'
$StatePath = Join-Path $LogRoot 'launcher_state.json'
$WatchdogStopPath = Join-Path $LogRoot 'watchdog.stop'
$LogPath = Join-Path $LogRoot ("AdaptiveAOI_stop_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-LauncherLog {
    param([string]$Level, [string]$Message)
    Add-Content -LiteralPath $LogPath -Value ("{0} [{1}] {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message) -Encoding UTF8
}

function Show-Message {
    param([string]$Text, [string]$Icon = 'Information')
    if ($NoDialog) { return }
    try {
        # Keep close notifications bounded for hidden shortcut launches too.
        $timeout = if ($env:ADAPTIVEAOI_DIALOG_TIMEOUT_SECONDS) { [Math]::Max(1, [int]$env:ADAPTIVEAOI_DIALOG_TIMEOUT_SECONDS) } else { 12 }
        $popupType = if ($Icon -eq 'Error') { 16 } else { 64 }
        [void]((New-Object -ComObject WScript.Shell).Popup($Text, $timeout, 'AdaptiveAOI 智能缺陷检测系统', $popupType))
    } catch { Write-LauncherLog 'WARN' "无法显示提示框：$($_.Exception.Message)" }
}

function Get-DescendantProcessIds {
    param([int]$RootPid)
    $queue = New-Object System.Collections.Generic.Queue[int]
    $ids = New-Object System.Collections.Generic.List[int]
    $queue.Enqueue($RootPid)
    while ($queue.Count -gt 0) {
        $parent = $queue.Dequeue()
        foreach ($child in (Get-CimInstance Win32_Process -Filter "ParentProcessId = $parent" -ErrorAction SilentlyContinue)) {
            $childPid = [int]$child.ProcessId
            if (-not $ids.Contains($childPid)) { [void]$ids.Add($childPid); $queue.Enqueue($childPid) }
        }
    }
    return $ids
}

function Stop-RecordedProcess {
    param([int]$ProcessId, [string]$Role, [string]$ExpectedStartedAt)
    if (-not $ProcessId) { return $false }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $details = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    $commandLine = [string]($details.CommandLine)
    $rootPattern = [regex]::Escape($SystemRoot.TrimEnd('\', '/')) + '(?:[\\/]|"|\s|$)'
    if ($commandLine -notmatch $rootPattern) {
        Write-LauncherLog 'WARN' "拒绝停止其他目录的 $Role 进程 PID=$ProcessId。"
        return $false
    }
    try {
        if (-not $ExpectedStartedAt -or $process.StartTime.ToUniversalTime().Ticks -ne ([DateTime]::Parse($ExpectedStartedAt)).ToUniversalTime().Ticks) {
            Write-LauncherLog 'WARN' "拒绝停止启动时间与记录不符的 $Role 进程 PID=$ProcessId。"
            return $false
        }
    } catch {
        Write-LauncherLog 'WARN' "无法确认 $Role 进程的启动时间，未停止 PID=$ProcessId。"
        return $false
    }
    $valid = if ($Role -eq 'watchdog') {
        $process.ProcessName -match '^powershell(?:\.exe)?$' -and $commandLine -match 'AdaptiveAOI' -and $commandLine -match '\.ps1'
    } elseif ($Role -eq 'backend') {
        $process.ProcessName -match '^python(?:\.exe)?$' -and $commandLine -match '(run_system_api\.py|-m\s+backend)'
    } else {
        $process.ProcessName -match '^python(?:\.exe)?$' -and $commandLine -match 'streamlit'
    }
    if (-not $valid) {
        Write-LauncherLog 'WARN' "拒绝停止未通过命令行校验的 $Role 进程 PID=$ProcessId。"
        return $false
    }
    try {
        # Some restricted Windows sessions raise a NullReferenceException from
        # Stop-Process even for an accessible process.  The recorded PID has
        # already been verified as this launcher's Python child, so use the
        # .NET process handle directly and confirm its exit.
        $process.Kill()
        if (-not $process.WaitForExit(5000)) {
            throw "进程在 5 秒内未退出。"
        }
        Write-LauncherLog 'INFO' "已停止launchers记录的 $Role 进程 PID=$ProcessId。"
        return $true
    } catch {
        Write-LauncherLog 'WARN' "停止 $Role 进程 PID=$ProcessId 失败：$($_.Exception.Message)"
        return $false
    }
}

Write-LauncherLog 'INFO' '开始关闭由launchers创建的 AdaptiveAOI 服务。'
Set-Content -LiteralPath $WatchdogStopPath -Value (Get-Date).ToString('o') -Encoding UTF8
if (-not (Test-Path -LiteralPath $StatePath)) {
    $message = "没有找到launchers进程记录，未关闭任何进程。`r`n日志：$LogPath"
    Write-LauncherLog 'INFO' $message
    Show-Message -Text $message
    exit 0
}

try {
    $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not [string]::Equals([System.IO.Path]::GetFullPath([string]$state.project_root), [System.IO.Path]::GetFullPath($SystemRoot), [System.StringComparison]::OrdinalIgnoreCase)) {
        throw '进程记录属于其他项目目录，未关闭任何进程。'
    }
    $stopped = $false
    if ($state.watchdog.started_by_launcher) { $stopped = (Stop-RecordedProcess -ProcessId ([int]$state.watchdog.pid) -Role 'watchdog' -ExpectedStartedAt ([string]$state.watchdog.process_started_at)) -or $stopped }
    Start-Sleep -Milliseconds 500
    if ($state.frontend.started_by_launcher) { $stopped = (Stop-RecordedProcess -ProcessId ([int]$state.frontend.pid) -Role 'frontend' -ExpectedStartedAt ([string]$state.frontend.process_started_at)) -or $stopped }
    if ($state.backend.started_by_launcher) { $stopped = (Stop-RecordedProcess -ProcessId ([int]$state.backend.pid) -Role 'backend' -ExpectedStartedAt ([string]$state.backend.process_started_at)) -or $stopped }
    if ($stopped) {
        $message = "AdaptiveAOI launchers创建的服务已发送关闭请求。`r`n日志：$LogPath"
        Write-LauncherLog 'INFO' $message
        Show-Message -Text $message
    } else {
        $message = "未发现仍在运行的launchers自有服务，未影响其他 Python 或 Node 进程。`r`n日志：$LogPath"
        Write-LauncherLog 'INFO' $message
        Show-Message -Text $message
    }
} catch {
    $message = "关闭 AdaptiveAOI 时发生错误：$($_.Exception.Message)`r`n日志：$LogPath"
    Write-LauncherLog 'ERROR' $message
    Show-Message -Text $message -Icon Error
    exit 1
}
