@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE=python"
if exist "venv\Scripts\python.exe" set "PY_EXE=venv\Scripts\python.exe"

"%PY_EXE%" "sniper_bot\main.py" ^
  --config-path "sniper_bot\sniper_bot_config.json" ^
  --local-paper ^
  %*

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Local live-paper runner exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
