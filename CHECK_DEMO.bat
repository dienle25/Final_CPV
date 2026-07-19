@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua co moi truong Python. Hay chay scripts\setup_windows.ps1 truoc.
  pause
  exit /b 1
)
echo Dang kiem tra model, roster, anh khuon mat va suy luan that...
".venv\Scripts\python.exe" "scripts\preflight.py"
set "CHECK_EXIT=%ERRORLEVEL%"
echo.
if "%CHECK_EXIT%"=="0" (
  echo KIEM TRA DAT. Co the chay RUN_CLASSROOM_DEMO.bat
) else (
  echo KIEM TRA KHONG DAT. Xem cac dong LOI o tren.
)
pause
exit /b %CHECK_EXIT%
