[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = if ($Dev) { "requirements-dev.txt" } else { "requirements-classroom.txt" }

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
    $Created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($Version in @("3.12", "3.11")) {
            & py "-$Version" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3, 11), (3, 12)] else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Tao moi truong Python $Version..."
                & py "-$Version" -m venv .venv
                if ($LASTEXITCODE -ne 0) { throw "Khong tao duoc .venv bang Python $Version." }
                $Created = $true
                break
            }
        }
    }
    if (-not $Created -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3, 11), (3, 12)] else 1)"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Tao moi truong Python..."
            & python -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw "Khong tao duoc .venv." }
            $Created = $true
        }
    }
    if (-not $Created) {
        throw "Can cai Python 3.11 hoac 3.12 (64-bit), sau do chay lai script. Tai: https://www.python.org/downloads/windows/"
    }
}

Write-Host "Cap nhat pip va cai runtime demo..."
Invoke-NativeChecked $VenvPython -m pip install --upgrade pip setuptools wheel
Invoke-NativeChecked $VenvPython -m pip install -r $Requirements

if (-not $SkipPreflight) {
    Write-Host "Kiem tra truoc khi chay..."
    Invoke-NativeChecked $VenvPython scripts\preflight.py
}

Write-Host ""
Write-Host "Cai dat hoan tat. Chay: .\scripts\run_classroom_demo.ps1"

