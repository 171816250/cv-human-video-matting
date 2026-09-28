@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project environment missing: "%~dp0.venv\Scripts\python.exe"
    echo Create .venv and install the documented dependencies first. Nothing was installed.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0scripts\start_demo.py" %*
exit /b %errorlevel%
