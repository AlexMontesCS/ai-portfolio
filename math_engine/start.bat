@echo off
echo ============================================================
echo STARTING MATH ENGINE - FULL STACK
echo ============================================================

echo.
echo Starting Backend API Server...
cd backend
start cmd /k "py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend Development Server...
cd ..\frontend
start cmd /k "npm run dev"

echo.
echo ============================================================
echo MATH ENGINE STARTED!
echo Backend API: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Frontend UI: http://localhost:5173
echo ============================================================
echo.
echo Press any key to continue...
pause >nul
