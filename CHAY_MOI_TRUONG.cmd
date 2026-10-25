@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-highway\Scripts\python.exe" goto missing
".venv-highway\Scripts\python.exe" -c "import highway_env, gymnasium, pygame" >nul 2>&1
if errorlevel 1 goto missing
".venv-highway\Scripts\python.exe" -m highway_project.run --paused %*
if errorlevel 1 pause
exit /b
:missing
echo Chua cai moi truong. Chay CAI_DAT_MOI_TRUONG.cmd truoc.
pause
exit /b 1
