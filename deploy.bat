@echo off
REM Resume Relevance Check System - Deployment Script
REM =================================================

echo 🚀 Resume Relevance Check System - Deployment
echo ===============================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ❌ Virtual environment not found!
    echo Please run setup first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo ✅ Activating virtual environment...
call venv\Scripts\activate

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found!
    echo Please create .env file with your GEMINI_API_KEY
    echo Example:
    echo   GEMINI_API_KEY=your_api_key_here
    echo.
    echo Continuing with demo mode...
)

REM Check dependencies
echo 🔍 Checking dependencies...
python check_deps.py

if %errorlevel% neq 0 (
    echo ❌ Dependency check failed!
    pause
    exit /b 1
)

REM Find available port
echo 🔍 Finding available port...
for /l %%p in (8501,1,10) do (
    netstat -ano | findstr :%%p >nul 2>&1
    if errorlevel 1 (
        set PORT=%%p
        goto :found_port
    )
)
echo ❌ No available ports found!
pause
exit /b 1

:found_port
echo ✅ Using port %PORT%

REM Start the application
echo.
echo 🎯 Starting Resume Relevance Check System...
echo 📱 Access at: http://localhost:%PORT%
echo 🛑 Press Ctrl+C to stop
echo.

streamlit run app.py --server.port=%PORT% --server.headless=true

REM Deactivate virtual environment
call venv\Scripts\deactivate