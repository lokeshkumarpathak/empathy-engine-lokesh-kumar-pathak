@echo off
echo ========================================
echo   EMPATHY ENGINE - Quick Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Create virtual environment
echo [*] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo [*] Installing dependencies (this may take 2-3 minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo [*] Creating directories...
if not exist "templates" mkdir templates
if not exist "audio_output" mkdir audio_output

echo.
echo [OK] Setup complete!
echo.
echo Next steps:
echo    1. Make sure index.html is in the templates\ folder
echo    2. Run: python app.py
echo    3. Open: http://localhost:5000
echo.
echo Happy hacking!
pause