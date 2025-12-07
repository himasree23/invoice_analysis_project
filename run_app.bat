@echo off
REM Quick start script for Invoice Analysis App

echo ============================================================
echo STARTING INVOICE ANALYSIS APPLICATION
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo.
echo Your browser will open automatically at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ============================================================
echo.

REM Run the Streamlit app
streamlit run app.py

REM If streamlit command fails
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to start application
    echo Please make sure all dependencies are installed
    echo Run: pip install -r requirements.txt
    pause
    exit /b 1
)