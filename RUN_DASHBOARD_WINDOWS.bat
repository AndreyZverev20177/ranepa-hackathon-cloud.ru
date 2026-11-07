@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Cloud Digest editor panel, local-only. Run from any current directory.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
    echo [INFO] .venv not found. Using Python from PATH.
)

"%PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed in this Python environment.
    echo Install: "%PYTHON%" -m pip install -e ".[ui]"
    exit /b 1
)

echo Cloud Digest UI: http://127.0.0.1:8501
echo To stop, press Ctrl+C in this window.
"%PYTHON%" -m streamlit run "ui\app.py" --server.address 127.0.0.1 --server.port 8501
exit /b %errorlevel%
