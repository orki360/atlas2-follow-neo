@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0run_update10_tests.py"
set "test_result=%errorlevel%"
pause
exit /b %test_result%
