@echo off
echo ============================================================
echo   CREDIT SCORING AI - SETUP ^& LAUNCH
echo   CodeAlpha ML Internship Project
echo ============================================================
echo.

REM Check for Python
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/3] Training models (downloads dataset automatically)...
python train_model.py
if errorlevel 1 (
    echo [ERROR] Training failed. Check error above.
    pause
    exit /b 1
)

echo.
echo [3/3] Launching Streamlit app...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop the server.
echo.
streamlit run app.py

pause
