@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-highway\Scripts\python.exe" goto missing
if not exist "highway_project\models\dqn_model.pth" (
    echo Chưa có mô hình DQN đã huấn luyện. Đang khởi động huấn luyện mô hình mới...
    ".venv-highway\Scripts\python.exe" -m highway_project.train_dqn
    if errorlevel 1 goto failed
)
echo Khởi chạy xe tự lái với mô hình Deep Q-Network (PyTorch)...
".venv-highway\Scripts\python.exe" -m highway_project.run --algo dqn --mode agent --auto-reset %*
if errorlevel 1 goto failed
exit /b

:missing
echo Chưa tìm thấy môi trường ảo. Hãy chạy CAI_DAT_MOI_TRUONG.cmd trước.
pause
exit /b 1

:failed
echo Có lỗi xảy ra trong quá trình thực thi.
pause
exit /b 1

