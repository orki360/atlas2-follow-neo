@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" goto missing
rem Always give HARD a real console HWND, including when launched from Windows Terminal.
rem The scoped marker is inherited by the child and prevents recursive relaunches.
if "%FOLLOW_NEO_CONSOLE_HOSTED%"=="1" goto run
set "FOLLOW_NEO_CONSOLE_HOSTED=1"
start "" /wait "%SystemRoot%\System32\conhost.exe" "%ComSpec%" /d /c call "%~f0" %*
exit /b %errorlevel%
:run
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
