@echo off
setlocal
echo === Python launcher ===
py -0p 2>nul
if errorlevel 1 echo py launcher not available or broken

echo.
echo === py -3 version ===
py -3 --version 2>&1

echo.
echo === python version ===
python --version 2>&1

echo.
echo === where python ===
where python 2>&1

echo.
echo === common install dirs ===
dir "%LOCALAPPDATA%\Programs\Python" /b 2>nul
dir "C:\Program Files\Python*" /b 2>nul
dir "C:\Program Files (x86)\Python*" /b 2>nul

echo.
echo If no Python 3.10+ is shown above, install Python 3.10+ and check Add python.exe to PATH.
pause
