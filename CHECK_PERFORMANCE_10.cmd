@echo off
setlocal
cd /d "%~dp0"
echo Offline 130-second benchmark. Close the GUI first. No drone connection.
"%~dp0.venv\Scripts\python.exe" "%~dp0benchmark_update10.py" --seconds 130 --compute Auto --profile Auto %*
set "BENCH10_EXIT=%ERRORLEVEL%"
pause
exit /b %BENCH10_EXIT%
