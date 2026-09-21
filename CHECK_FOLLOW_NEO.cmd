@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" goto missing
"%~dp0.venv\Scripts\python.exe" "%~dp0run_lab.py" --check
set "check_result=%errorlevel%"
pause
exit /b %check_result%
:missing
echo Run SETUP_FOLLOW_NEO.cmd first.
pause
exit /b 1
