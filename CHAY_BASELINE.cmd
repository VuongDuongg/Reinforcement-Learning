  @echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-highway\Scripts\python.exe" goto missing
echo Dang khoi dong AI Rule-based Baseline (Tap luat an toan Heuristic)...
".venv-highway\Scripts\python.exe" -m highway_project.run --algo baseline --mode agent --auto-reset %*
if errorlevel 1 goto failed
exit /b
:missing
echo Chua cai moi truong. Chay CAI_DAT_MOI_TRUONG.cmd truoc.
:failed
pause
exit /b 1
