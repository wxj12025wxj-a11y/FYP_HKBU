@echo off
REM ===================================================================
REM MUSDB18-HQ test set song-major sweep -- launched by Windows Task Scheduler
REM
REM Why not nohup / run_in_background:
REM   2026-09-21 measured: a background process started by the agent session is
REM   HARD-KILLED when that session turn ends (TerminateProcess -- the finally
REM   block never runs, leaving a stale _test_sweep.lock behind).
REM   13:33:52 start -> 13:52:10 killed, only 18 minutes. No sleep, no crash event.
REM   Under Task Scheduler the process is owned by the service, fully decoupled.
REM
REM Output: redirect straight to a file. Do NOT use tee -- tee dies with EPIPE
REM   when the session pipe closes and takes python down via SIGPIPE.
REM
REM NOTE: keep this file ASCII-only. Chinese comments get mis-decoded as GBK by
REM   cmd.exe and can eat the following line (here it silently killed the cd).
REM ===================================================================
cd /d E:\FYP_HKBU

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

set PY=C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe
set OUT=E:\FYP_HKBU\tools\_logs\_sweep_detached.out

echo ===== launch %DATE% %TIME% ===== >> "%OUT%"
"%PY%" -X utf8 tools\bench_musdb_test_songmajor.py --models all --songs all --timeout 180 >> "%OUT%" 2>&1
echo ===== exit %ERRORLEVEL% %DATE% %TIME% ===== >> "%OUT%"
