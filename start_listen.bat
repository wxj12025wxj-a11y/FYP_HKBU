@echo off
REM ============================================================
REM  Launch the local listening page for the MUSDB18 stem A/B test.
REM  A tiny static server (Range-enabled) is required because the
REM  browser must be able to seek inside 430 s lossless files.
REM  Double-click this file, then use the URL it prints.
REM ============================================================
setlocal
set PY=C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe
set ROOT=E:\FYP_HKBU
set PORT=8123
set PAGE=http://127.0.0.1:%PORT%/04_reports/separation/html/listen_compare.html

if not exist "%PY%" set PY=python

echo Starting listening server on port %PORT% ...
start "FYP listen server" "%PY%" "%ROOT%\tools\_listen_server.py" --root "%ROOT%" --port %PORT%
timeout /t 2 /nobreak >nul
start "" "%PAGE%"
echo.
echo   Page : %PAGE%
echo   Close the "FYP listen server" window to stop the server.
echo.
endlocal
