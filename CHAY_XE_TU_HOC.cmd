@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-highway\Scripts\python.exe" goto missing
if not exist "highway_project\models\q_learning.npz" (
    ".venv-highway\Scripts\python.exe" -m highway_project.train
    if errorlevel 1 goto failed
)
".venv-highway\Scripts\python.exe" -m highway_project.run --mode agent --auto-reset %*
if errorlevel 1 goto failed
exit /b
:missing
echo Chay CAI_DAT_MOI_TRUONG.cmd truoc.
:failed
pause
exit /b 1
