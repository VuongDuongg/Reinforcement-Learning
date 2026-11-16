@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ==============================================================================
echo       HE THONG XE TU LAI NE VAT CAN CAO TOC (REINFORCEMENT LEARNING)
echo ==============================================================================

:: 1. Kiem tra moi truong ao Python
if not exist ".venv-highway\Scripts\python.exe" (
    echo.
    echo [THONG BAO] Chua tim thay moi truong ao Python .venv-highway!
    echo Vui long chay file CAI_DAT_MOI_TRUONG.cmd truoc de cai dat thu vien.
    echo.
    pause
    exit /b 1
)

:: 2. Kiem tra checkpoint mo hinh DQN (neu chua co se tu dong huan luyen)
if not exist "highway_project\models\dqn_model.pth" (
    echo.
    echo [KHOI TAO] Chua co checkpoint DQN. Dang tu dong huan luyen mo hinh moi...
    ".venv-highway\Scripts\python.exe" -m highway_project.train_dqn
    if errorlevel 1 goto failed
)

:: 3. Kiem tra checkpoint mo hinh Q-learning (neu chua co se tu dong huan luyen)
if not exist "highway_project\models\q_learning.npz" (
    echo.
    echo [KHOI TAO] Chua co bang Q-learning. Dang tu dong khoi tao mo hinh Q-learning...
    ".venv-highway\Scripts\python.exe" -m highway_project.train
    if errorlevel 1 goto failed
)

:: 4. Khoi dong truc tiep giao dien mo phong Pygame (Day du tinh nang)
echo.
echo [DANG KHOI DONG] Dang mo giao dien mo phong Pygame...
if "%~1"=="" (
    ".venv-highway\Scripts\python.exe" -m highway_project.run --algo dqn --mode agent --auto-reset
) else (
    ".venv-highway\Scripts\python.exe" -m highway_project.run %*
)
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo ==============================================================================
echo [LOI] Chuong trinh gap su co trong qua trinh thuc thi.
echo ==============================================================================
pause
exit /b 1
