@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP_FOLLOW_NEO.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_lab.py --check --compute GPU
set "FOLLOW_RESULT=%ERRORLEVEL%"
pause
exit /b %FOLLOW_RESULT%
