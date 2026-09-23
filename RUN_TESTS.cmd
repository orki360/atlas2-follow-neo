@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" goto missing
"%~dp0.venv\Scripts\python.exe" "%~dp0run_update9_tests.py"
set "test_result=%errorlevel%"
pause
exit /b %test_result%
:missing
echo Run SETUP_FOLLOW_NEO.cmd first.
pause
exit /b 1
