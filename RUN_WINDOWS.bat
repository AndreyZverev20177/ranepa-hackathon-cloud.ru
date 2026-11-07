@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Usage: RUN_WINDOWS.bat [story|baseline|semantic]
rem New outputs are written to results/ to preserve partner_results/ history.
set "MODE=%~1"
if "%MODE%"=="" set "MODE=story"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Missing virtual environment. Run: python -m venv .venv
    echo Then: .\.venv\Scripts\python.exe -m pip install -e .
    exit /b 1
)

set "INPUT_FILE="
for %%F in ("partner_data\*26-31.08.json") do (
    if exist "%%~fF" set "INPUT_FILE=%%~fF"
)
if not defined INPUT_FILE (
    echo Partner JSON export not found in partner_data.
    exit /b 1
)

set "SCORE_ARGS=--scoring story-v1"
set "CLUSTER_ARGS="
if /I "%MODE%"=="baseline" (
    set "SCORE_ARGS=--scoring baseline"
) else if /I "%MODE%"=="story" (
    rem Keep story score, use baseline clustering without downloading a model.
) else if /I "%MODE%"=="semantic" (
    set "CLUSTER_ARGS=--clustering semantic --embedding-device cpu"
) else (
    echo Unknown mode: %MODE%. Allowed: story, baseline, semantic.
    exit /b 2
)

echo Running Cloud Digest: %MODE%
for %%D in (26 27 28 29 30 31) do (
    echo Date: 2026-08-%%D
    "%PYTHON%" -m cloud_digest --partner-export --input "%INPUT_FILE%" --date 2026-08-%%D --output-dir "results\%MODE%\%%D" %SCORE_ARGS% %CLUSTER_ARGS%
    if errorlevel 1 goto :failed
)
echo Complete. Open: results\%MODE%
exit /b 0

:failed
echo Processing failed. See the error above.
exit /b 1
