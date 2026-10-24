@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed
echo V18 shared profile launcher
echo Close all other Follow NEO windows before using this launcher.
echo Each launch restores the shared experiment values and backs up existing settings.
echo Phone IP, compute backend and GPU adapter are kept.
echo.
if not exist "%~dp0.venv\Scripts\python.exe" goto missing
if not exist "%~dp0load_v18_profile.py" goto incomplete
if not exist "%~dp0profiles\v18_shared.json" goto incomplete
if not exist "%~dp0RUN_FOLLOW_NEO.cmd" goto incomplete
"%~dp0.venv\Scripts\python.exe" -B "%~dp0load_v18_profile.py"
if errorlevel 1 goto failed
echo.
echo Profile loaded successfully. Starting Follow NEO...
call "%~dp0RUN_FOLLOW_NEO.cmd"
exit /b %errorlevel%
:missing
echo Run SETUP_FOLLOW_NEO.cmd first, then use this launcher again.
pause
exit /b 1
:incomplete
echo The v18 package is incomplete. Download and extract the updated v18 branch.
pause
exit /b 1
:failed
echo Profile loading failed. Follow NEO was not started. Copy the error above.
pause
exit /b 1
