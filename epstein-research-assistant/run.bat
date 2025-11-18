@echo off
REM Epstein Files Research Assistant - Launch Script (Windows)

echo 🔍 Epstein Files Investigative Research Assistant
echo ==================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Creating virtual environment...
    python -m venv .venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    echo ✅ Dependencies installed
)

REM Check if .env exists
if not exist ".env" (
    echo ⚠️  .env file not found!
    echo Copying from .env.example...
    copy .env.example .env
    echo ⚠️  Please edit .env and add your GEMINI_API_KEY
    echo.
    pause
)

REM Check if data directory exists
if not exist "data" (
    echo Creating data directory...
    mkdir data
)

REM Launch the app
echo.
echo 🚀 Launching application...
echo Opening browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run src\app.py

pause
