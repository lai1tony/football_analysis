@echo off
setlocal EnableExtensions
pushd "%~dp0"

set "PORT=5050"
set "APP_URL=http://127.0.0.1:5050/?_v=source-handicap-top3-20260618"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "APP_PY=%CD%\data\app.py"

if exist "%PY_EXE%" goto HAVE_VENV

echo Project virtual environment not found. Running project installer...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\installer\install.ps1"
if errorlevel 1 goto INSTALL_FAILED

if exist "%PY_EXE%" goto HAVE_VENV
echo Project virtual environment is still missing: %PY_EXE%
goto FAILED

:HAVE_VENV
echo Checking Python dependencies...
"%PY_EXE%" -c "import flask, requests, bs4, openpyxl" >nul 2>nul
if not errorlevel 1 goto DEPS_OK

echo Missing Python dependencies. Installing requirements...
"%PY_EXE%" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 goto DEPS_FAILED

"%PY_EXE%" -c "import flask, requests, bs4, openpyxl" >nul 2>nul
if errorlevel 1 goto DEPS_FAILED

:DEPS_OK
echo Stopping any existing service on port %PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do taskkill /PID %%P /F >nul 2>nul

echo Starting Football Analysis project service...
echo URL: %APP_URL%
echo Keep this window open while using the app. Press Ctrl+C to stop.
echo.
start "" "%APP_URL%"
"%PY_EXE%" "%APP_PY%"
pause
exit /b 0

:INSTALL_FAILED
echo Project installer failed. Please install or repair Python 3.10+ and Node.js LTS, then run start_app.bat again.
echo You can diagnose Python discovery with diagnose_python.bat
goto FAILED

:DEPS_FAILED
echo Failed to install or import project dependencies from requirements.txt.
echo Try running: "%PY_EXE%" -m pip install -r "%CD%\requirements.txt"
goto FAILED

:FAILED
pause
exit /b 1
