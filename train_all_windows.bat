@echo off
REM Re-downloads datasets, retrains every model and regenerates reports\EVALUATION_REPORT.md
cd /d "%~dp0"
.venv\Scripts\python.exe train_all.py %*
pause
