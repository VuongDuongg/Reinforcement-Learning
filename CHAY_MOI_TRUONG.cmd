@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-highway\Scripts\python.exe" goto missing

".venv-highway\Scripts\python.exe" -m highway_project.launcher %*
if errorlevel 1 pause
exit /b

:missing
echo [ERROR] Virtual environment not found. Please run CAI_DAT_MOI_TRUONG.cmd first.
pause
exit /b 1
