@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE=python"
if exist "venv\Scripts\python.exe" set "PY_EXE=venv\Scripts\python.exe"

"%PY_EXE%" "sniper_bot\sim_live_engine.py" ^
  --config-path "sniper_bot\sniper_bot_config.json" ^
  --timeframe "1h" ^
  --lookback-hours 168 ^
  --max-symbols 40 ^
  --progress-every-hours 12 ^
  --send-telegram ^
  --quiet-scanner ^
  %*

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Dry-run finished with exit code %EXIT_CODE%.
exit /b %EXIT_CODE%
