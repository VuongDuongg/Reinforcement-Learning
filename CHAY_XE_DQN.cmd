@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist ".venv-highway\Scripts\python.exe" goto missing

if not exist "highway_project\models\dqn_model.pth" (
    echo [THONG BAO] Chua co checkpoint DQN. Dang khoi dong huan luyen mo hinh moi...
    ".venv-highway\Scripts\python.exe" -m highway_project.train_dqn
    if errorlevel 1 goto failed
)

".venv-highway\Scripts\python.exe" -m highway_project.launcher %*
if errorlevel 1 goto failed
exit /b

:missing
echo [ERROR] Virtual environment not found. Please run CAI_DAT_MOI_TRUONG.cmd first.
pause
exit /b 1

:failed
echo [ERROR] Execution failed.
pause
exit /b 1
