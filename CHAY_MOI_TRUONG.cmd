@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist ".venv-highway\Scripts\python.exe" goto missing

:: Nếu người dùng truyền tham số từ dòng lệnh (ví dụ: CHAY_MOI_TRUONG.cmd --lanes 5), chạy trực tiếp
if not "%~1"=="" (
    ".venv-highway\Scripts\python.exe" -m highway_project.run %*
    goto end
)

:menu
cls
echo ==============================================================================
echo        MÔ PHỎNG XE TỰ HÀNH NÉ VẬT CẢN (REINFORCEMENT LEARNING)
echo ==============================================================================
echo.
echo  Chọn kịch bản muốn chạy:
echo.
echo   [1] Mặc định: Cao tốc 3 làn · Mật độ Bình thường (20 xe) · Tốc độ 72 km/h (DQN)
echo   [2] Kịch bản 1: Cao tốc hẹp 2 làn · Mật độ Thưa thớt (8 xe) · Tốc độ 90 km/h
echo   [3] Kịch bản 2: Cao tốc mở rộng 5 làn · Mật độ Dày đặc (38 xe) · Dòng xe nhanh
echo   [4] Kịch bản 3: Đường vắng không có xe (4 làn, 90 km/h, chỉ né xe hỏng)
echo   [5] Kịch bản 4: Tự lái bằng bàn phím (Người điều khiển)
echo   [6] TỰ TÙY BIẾN CHI TIẾT (Chọn số làn, mật độ xe, tốc độ theo ý bạn)
echo   [0] Thoát
echo.
echo ==============================================================================
set /p choice="Nhập lựa chọn của bạn [1-6] (Mặc định: 1): "

if "%choice%"=="" set choice=1
if "%choice%"=="1" goto run_default
if "%choice%"=="2" goto run_s1
if "%choice%"=="3" goto run_s2
if "%choice%"=="4" goto run_s3
if "%choice%"=="5" goto run_keyboard
if "%choice%"=="6" goto custom_setup
if "%choice%"=="0" exit /b
echo Lựa chọn không hợp lệ, vui lòng chọn lại!
timeout /t 2 >nul
goto menu

:run_default
echo.
echo Đang khởi chạy kịch bản Mặc định...
".venv-highway\Scripts\python.exe" -m highway_project.run --algo dqn --mode agent --auto-reset
goto end

:run_s1
echo.
echo Đang khởi chạy kịch bản 1 (2 làn, Thưa thớt, 90 km/h)...
".venv-highway\Scripts\python.exe" -m highway_project.run --lanes 2 --traffic sparse --speed 90 --max-speed 120 --algo dqn --mode agent --auto-reset
goto end

:run_s2
echo.
echo Đang khởi chạy kịch bản 2 (5 làn, Dày đặc, 80 km/h, xe nhanh)...
".venv-highway\Scripts\python.exe" -m highway_project.run --lanes 5 --traffic dense --speed 80 --max-speed 110 --traffic-speed fast --algo dqn --mode agent --auto-reset
goto end

:run_s3
echo.
echo Đang khởi chạy kịch bản 3 (4 làn, Không có xe, 90 km/h)...
".venv-highway\Scripts\python.exe" -m highway_project.run --lanes 4 --traffic none --speed 90 --algo dqn --mode agent --auto-reset
goto end

:run_keyboard
echo.
echo Đang khởi chạy chế độ Người lái tự điều khiển bằng bàn phím...
".venv-highway\Scripts\python.exe" -m highway_project.run --mode keyboard
goto end

:custom_setup
cls
echo ==============================================================================
echo                      CẤU HÌNH TÙY BIẾN KỊCH BẢN
echo ==============================================================================
echo.
set /p lanes="1. Số làn đường cao tốc [2..6] (Mặc định: 3): "
if "%lanes%"=="" set lanes=3

echo.
echo  2. Chọn mật độ xe giao thông:
echo   [1] Thưa thớt (sparse: 8 xe)
echo   [2] Bình thường (normal: 20 xe - Mặc định)
echo   [3] Dày đặc (dense: 38 xe)
echo   [4] Không có xe (none: 0 xe)
set /p density_choice="Nhập lựa chọn [1-4] (Mặc định: 2): "
if "%density_choice%"=="1" set traffic=sparse
if "%density_choice%"=="3" set traffic=dense
if "%density_choice%"=="4" set traffic=none
if not defined traffic set traffic=normal

echo.
set /p speed="3. Vận tốc xuất phát xe của bạn theo km/h (Mặc định: 72): "
if "%speed%"=="" set speed=72

echo.
echo  4. Chọn tốc độ dòng xe xung quanh:
echo   [1] Chậm (slow: 36-65 km/h)
echo   [2] Bình thường (normal: 50-95 km/h - Mặc định)
echo   [3] Nhanh (fast: 75-115 km/h)
set /p tspd_choice="Nhập lựa chọn [1-3] (Mặc định: 2): "
if "%tspd_choice%"=="1" set traffic_speed=slow
if "%tspd_choice%"=="3" set traffic_speed=fast
if not defined traffic_speed set traffic_speed=normal

echo.
echo  5. Chọn chế độ điều khiển:
echo   [1] AI tự lái bằng Deep Q-Network (DQN - Mặc định)
echo   [2] Bạn tự lái bằng bàn phím
set /p mode_choice="Nhập lựa chọn [1-2] (Mặc định: 1): "
if "%mode_choice%"=="2" (
    set run_mode=keyboard
) else (
    set run_mode=agent
)

echo.
echo ==============================================================================
echo Đang khởi động mô phỏng với cấu hình:
echo  - Số làn đường: %lanes% làn
echo  - Mật độ xe: %traffic%
echo  - Vận tốc xuất phát: %speed% km/h
echo  - Dòng xe: %traffic_speed%
echo  - Chế độ: %run_mode%
echo ==============================================================================
echo.
".venv-highway\Scripts\python.exe" -m highway_project.run --lanes %lanes% --traffic %traffic% --speed %speed% --traffic-speed %traffic_speed% --algo dqn --mode %run_mode% --auto-reset

:end
if errorlevel 1 pause
exit /b

:missing
echo [LỖI] Chưa tìm thấy môi trường ảo. Hãy chạy CAI_DAT_MOI_TRUONG.cmd trước!
pause
exit /b 1
