@echo off
set PORT=5050
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
  echo Stopping process %%a on port %PORT%
  taskkill /PID %%a /F
)
