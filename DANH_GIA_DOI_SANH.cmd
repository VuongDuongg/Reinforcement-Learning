@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-highway\Scripts\python.exe" goto missing
echo =========================================================================
echo   TIEN HANH DANH GIA DOI SANH (BENCHMARKING) BASELINE VA CAC THUAT TOAN
echo =========================================================================
echo.
".venv-highway\Scripts\python.exe" -m highway_project.training.benchmark_all --episodes 15
if errorlevel 1 goto failed
echo.
echo Hoan thanh danh gia. File ket qua da duoc luu tai thu muc highway_project\results\
pause
exit /b 0

:missing
echo Chua cai moi truong. Chay CAI_DAT_MOI_TRUONG.cmd truoc.
:failed
pause
exit /b 1
