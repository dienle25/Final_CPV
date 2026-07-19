# Tên file được giữ để tương thích với tài liệu/bản nộp cũ.
# Script mới không xóa lịch sử; nó mở đúng giao diện demo lớp học.
[CmdletBinding()]
param(
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$Runner = (Resolve-Path (Join-Path $PSScriptRoot "run_classroom_demo.ps1")).Path
& $Runner -SkipPreflight:$SkipPreflight
if ($LASTEXITCODE -ne 0) {
    throw "Demo kết thúc với mã lỗi $LASTEXITCODE."
}

