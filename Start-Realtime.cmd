@echo off
setlocal
if not exist "%~dp0.venv-gpu\Scripts\python.exe" (
    echo GPU environment missing: "%~dp0.venv-gpu\Scripts\python.exe"
    echo See docs/REALTIME.md to prepare it. Nothing was installed.
    exit /b 1
)
"%~dp0.venv-gpu\Scripts\python.exe" -B "%~dp0scripts\start_realtime.py" %*
exit /b %errorlevel%
