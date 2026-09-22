@echo off
REM Smoke test: verify Task Scheduler -> venv python -> CUDA -> disk write chain.
REM Run this (~2 min) before committing to the ~8 hour full sweep.
REM ASCII only -- see note in _sweep_launch.bat.
cd /d E:\FYP_HKBU

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

set PY=C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe
set OUT=E:\FYP_HKBU\tools\_logs\_sweep_smoke.out

echo ===== smoke launch %DATE% %TIME% ===== > "%OUT%"
echo CWD=%CD% >> "%OUT%"
"%PY%" -X utf8 tools\bench_musdb_test_songmajor.py --models umx --songs 1:2 --timeout 180 >> "%OUT%" 2>&1
echo ===== smoke exit %ERRORLEVEL% %DATE% %TIME% ===== >> "%OUT%"
