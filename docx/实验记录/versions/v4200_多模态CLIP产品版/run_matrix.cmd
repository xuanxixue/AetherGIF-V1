@echo off
REM ============================================================
REM  AetherGIF v4200 - phase matrix test (directions 1 / 2 / 3)
REM  Usage: run_matrix.cmd [STEPS]
REM
REM  Direction 1 (COND_LOWRANK 16->64): lift the low-rank bottleneck.
REM  Direction 2 (PH_ABS_W_MODE): per-channel uniform weight for phase abs loss.
REM  Direction 3 (longer training)  : expressed as STEPS on every group.
REM  Direction 4 (bigger PH_ABS_W)  : deliberately NOT used (phabs2k lesson -
REM              weight 8 -> magnitude ~80 crushes every other loss item).
REM              Only the clamp boundary is used instead (IDEAL_CLAMP=pi).
REM
REM  Pure ASCII: self-referencing via %~dp0, no non-ASCII baked in.
REM  Runs sequentially - a 4GB GPU cannot run two trainings at once.
REM  Base ckpt: outputs_real\fix_r4\cond3_ckpt.pt (100 steps, wcos +0.5077)
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set STEPS=%1
if "%STEPS%"=="" set STEPS=200

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=C:\Python312\python.exe

REM ---- shared env (structure must match training) ----
set COND_GROUPS=5
set PHASE_ACT=arctan
set RES_LIMIT=0.5
set W2_NORM_MAX=3.0
set KAPPA_MAX=6.0
set FFT_FFTW=1
set HF_GUARD_W=1.0
set LO_FLOOR_W=0.5
set GUARD_W=1.0
set ACC_K=2
set PH_ABS_W=1.0
set FFT_W=1.0
set BAND_ABS_W=0.3
set IDEAL_CLAMP=3.141592653589793

echo ============================================================
echo   Phase matrix A..E  STEPS=%STEPS%
echo   Base ckpt: outputs_real\fix_r4\cond3_ckpt.pt (100 steps, wcos +0.5077)
echo   Safe-range 4: IDEAL_CLAMP=pi (not open), PH_ABS_W=1.0 (not 8.0)
echo   Direction 3 (longer training) = larger STEPS on every group
echo ============================================================
echo.

REM ---- Group A: lowrank=16, mode=norm (continuation baseline) ----
echo [A] lowrank=16 mode=norm
set OVERFIT_TAG=mx_A
set COND_LOWRANK=16
set PH_ABS_W_MODE=norm
set RESUME=outputs_real\fix_r4\cond3_ckpt.pt
set RESUME_R_OLD=
%PY% -u scripts\overfit_cond3.py %STEPS% > outputs_real\mx_A.log 2>&1

REM ---- Group B: lowrank=64 (direction1: lift low-rank bottleneck) ----
echo [B] lowrank=64 mode=norm
set OVERFIT_TAG=mx_B
set COND_LOWRANK=64
set PH_ABS_W_MODE=norm
set RESUME_R_OLD=16
%PY% -u scripts\overfit_cond3.py %STEPS% > outputs_real\mx_B.log 2>&1

REM ---- Group C: lowrank=16, mode=uniform (direction2: max high-freq grad) ----
echo [C] lowrank=16 mode=uniform
set OVERFIT_TAG=mx_C
set COND_LOWRANK=16
set PH_ABS_W_MODE=uniform
set RESUME_R_OLD=
%PY% -u scripts\overfit_cond3.py %STEPS% > outputs_real\mx_C.log 2>&1

REM ---- Group D: lowrank=64 + mode=uniform (combo of 1 and 2) ----
echo [D] lowrank=64 mode=uniform
set OVERFIT_TAG=mx_D
set COND_LOWRANK=64
set PH_ABS_W_MODE=uniform
set RESUME_R_OLD=16
%PY% -u scripts\overfit_cond3.py %STEPS% > outputs_real\mx_D.log 2>&1

REM ---- Group E: lowrank=16, mode=band ----
echo [E] lowrank=16 mode=band
set OVERFIT_TAG=mx_E
set COND_LOWRANK=16
set PH_ABS_W_MODE=band
set RESUME_R_OLD=
%PY% -u scripts\overfit_cond3.py %STEPS% > outputs_real\mx_E.log 2>&1

echo.
echo ============================================================
echo   Matrix done. Logs: outputs_real\mx_A.log .. mx_E.log
echo ============================================================
