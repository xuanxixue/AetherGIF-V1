@echo off
REM ============================================================
REM  AetherGIF v4200 - Multimodal CLIP Product - Full Training
REM  Pure ASCII: self-referencing via %~dp0, no non-ASCII baked in
REM  (cmd parses this file as GBK and would mangle Chinese paths).
REM  Output is teed to logs_train.txt so a post-mortem is possible
REM  even if the window is closed.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set LOG=%CD%\logs_train.txt

echo ============================================================
echo   AetherGIF v4200  -  Multimodal CLIP Full Training
echo   Budget 7 days max | Early stop ON (4 criteria)
echo   LR 3e-5 warmup300 to 3e-4 cosine to 1e-5
echo   Closing this window STOPS training (ckpt saved every 200 steps)
echo ============================================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

C:\Python312\python.exe -u scripts\train_multimodal.py %* > "%LOG%" 2>&1
set RC=%ERRORLEVEL%

type "%LOG%"
echo.
echo ============================================================
echo   Process exited. RC=%RC%
echo   Log: %LOG%
echo   Weights: outputs_real\ckpt\
echo ============================================================
pause