@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" goto missing
"%~dp0.venv\Scripts\python.exe" "%~dp0run_lab.py" %*
if errorlevel 1 goto failed
exit /b 0
:missing
echo Run SETUP_FOLLOW_NEO.cmd first.
pause
exit /b 1
:failed
echo The program stopped with an error. See logs\last_error.txt
pause
exit /b 1
