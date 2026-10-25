@echo off
setlocal
cd /d "%~dp0"
if not exist ".tmp-highway" mkdir ".tmp-highway"
set "TEMP=%CD%\.tmp-highway"
set "TMP=%TEMP%"
if not exist ".venv-highway\Scripts\python.exe" (
    python -m venv .venv-highway
    if errorlevel 1 goto failed
)
".venv-highway\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
".venv-highway\Scripts\python.exe" -m pip install -r highway_project\requirements.txt
if errorlevel 1 goto failed
echo.
echo Cai xong. Mo CHAY_MOI_TRUONG.cmd de xem xe.
pause
exit /b 0
:failed
echo.
echo Cai dat chua thanh cong. Kiem tra Python va ket noi Internet o thong bao tren.
pause
exit /b 1
