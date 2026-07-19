[CmdletBinding()]
param(
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Lenh that bai (exit $LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Chua co .venv. Hay chay .\scripts\setup_windows.ps1 truoc."
}

if (-not $SkipPreflight) {
    Invoke-NativeChecked $VenvPython scripts\preflight.py
}

Write-Host "Mo giao dien demo tai http://localhost:8501"
Invoke-NativeChecked $VenvPython -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501

