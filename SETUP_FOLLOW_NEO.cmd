@echo off
setlocal
cd /d "%~dp0"
py -3.12 "%~dp0setup_env.py"
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click RUN_FOLLOW_NEO.cmd
pause
exit /b 0
:failed
echo.
echo Setup failed. Copy the output above. Python 3.12 x64 is required.
pause
exit /b 1
