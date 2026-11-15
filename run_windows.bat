@echo off
REM Starts the web app at http://127.0.0.1:5000 and opens the browser.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run setup_windows.bat first. & pause & exit /b 1)
start "" http://127.0.0.1:5000
.venv\Scripts\python.exe run.py
pause
