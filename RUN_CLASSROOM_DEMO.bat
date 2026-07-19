@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_classroom_demo.ps1"
if errorlevel 1 (
  echo.
  echo Demo khong khoi dong duoc. Xem thong bao loi o tren.
  pause
  exit /b 1
)
endlocal

