@echo off
REM Invoice Analysis Project - Windows Setup Script
REM Run this file to automatically set up the project

echo ============================================================
echo INVOICE ANALYSIS PROJECT - WINDOWS SETUP
echo ============================================================
echo.

REM Check Python installation
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or 3.11 from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo OK - Python is installed
echo.

REM Create virtual environment
echo [2/7] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo OK - Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    echo If using PowerShell, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    pause
    exit /b 1
)
echo OK - Virtual environment activated
echo.

REM Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo [5/7] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo OK - All dependencies installed
echo.

REM Create directory structure
echo [6/7] Creating directory structure...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "models" mkdir "models"
if not exist "src" mkdir "src"
echo OK - Directory structure created
echo.

REM Create __init__.py files
echo [7/7] Creating Python package files...
type nul > src\__init__.py
echo OK - Package files created
echo.

echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo 1. Keep this terminal open (virtual environment is active)
echo 2. Run: streamlit run app.py
echo 3. Your browser will open with the application
echo.
echo To activate the environment in the future:
echo    venv\Scripts\activate
echo.
echo To deactivate:
echo    deactivate
echo.
echo ============================================================
pause