@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  echo Create it: python -m venv .venv
  exit /b 1
)
".venv\Scripts\python.exe" "tools\run_compare.py" %*
exit /b %errorlevel%
