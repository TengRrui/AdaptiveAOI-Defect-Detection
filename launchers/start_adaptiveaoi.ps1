[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$currentName = [System.IO.Path]::GetFileName($PSCommandPath)
$launcher = Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter '*.ps1' |
    Where-Object { $_.Name -ne $currentName } |
    Where-Object { (Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8) -match 'function Resolve-DataRoot' } |
    Select-Object -First 1

if (-not $launcher) {
    Write-Error 'AdaptiveAOI launcher implementation was not found.'
    exit 11
}

& $launcher.FullName
exit $LASTEXITCODE
