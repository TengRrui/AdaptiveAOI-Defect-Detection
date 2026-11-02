[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$LauncherRoot = $PSScriptRoot
$SystemRoot = Split-Path -Parent $LauncherRoot
$ShortcutPath = Join-Path $SystemRoot 'AdaptiveAOI智能缺陷检测系统.lnk'
$TargetPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$IconPath = Join-Path $LauncherRoot 'assets\AdaptiveAOI.ico'
$Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $LauncherRoot 'start.ps1') + '"'

foreach ($path in @($TargetPath, $IconPath, $SystemRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "创建快捷方式失败，必要路径不存在：$path" }
}

if (Test-Path -LiteralPath $ShortcutPath) {
    throw "目标快捷方式已存在，拒绝覆盖：$ShortcutPath。请先按项目归档规则完成独立备份和校验后再重新创建。"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $TargetPath
$shortcut.Arguments = $Arguments
$shortcut.WorkingDirectory = $SystemRoot
$shortcut.IconLocation = "$IconPath,0"
$shortcut.WindowStyle = 7
$shortcut.Description = '启动 AdaptiveAOI 智能缺陷检测系统'
$shortcut.Hotkey = ''
$shortcut.Save()

$verified = $shell.CreateShortcut($ShortcutPath)
$result = [pscustomobject]@{
    shortcut_path = $ShortcutPath
    target_path = $verified.TargetPath
    arguments = $verified.Arguments
    working_directory = $verified.WorkingDirectory
    icon_location = $verified.IconLocation
    description = $verified.Description
    target_exists = Test-Path -LiteralPath $verified.TargetPath
    launcher_argument_valid = $verified.Arguments -match [regex]::Escape((Join-Path $LauncherRoot 'start.ps1'))
    icon_exists = Test-Path -LiteralPath $IconPath
}
$result | ConvertTo-Json -Depth 3
if (-not $result.target_exists -or -not $result.icon_exists -or -not $result.launcher_argument_valid) { exit 1 }
