@echo off
REM One-time setup on Windows: creates a virtual environment and installs dependencies.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
python -m venv .venv || (echo Failed to create virtual environment & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Dependency installation failed & pause & exit /b 1)
echo.
echo Setup complete. Start the app with run_windows.bat
pause
