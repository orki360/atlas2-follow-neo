@echo off
rem Compatibility alias. The normal launcher now supplies the dedicated console.
call "%~dp0RUN_FOLLOW_NEO.cmd" --runtime process %*
exit /b %errorlevel%
