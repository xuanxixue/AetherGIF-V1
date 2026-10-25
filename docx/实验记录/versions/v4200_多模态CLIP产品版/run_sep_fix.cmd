@echo off
REM ============================================================
REM  AetherGIF v4200 - loss_sep sign fix smoke test (32nd pitfall)
REM  Usage: run_sep_fix.cmd [STEPS]
REM
REM  Base ckpt: outputs_real\fix_r4\cond3_ckpt.pt (same start as mx_A,
REM  fair comparison). loss_sep sign was inverted: clamp(max=0) paid
REM  a NEGATIVE reward for "no inter-band diff + intra-band noise".
REM  After the fix it is a positive penalty (relu), so gen must stop
REM  dominating the total loss at -9.77.
REM
REM  Pure ASCII per project rule. Runs sequentially on 4GB GPU.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set STEPS=%1
if "%STEPS%"=="" set STEPS=50

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=C:\Python312\python.exe

REM ---- shared env (structure must match run_matrix.cmd) ----
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

REM ---- this run ----
set OVERFIT_TAG=fix_sep_50
set COND_LOWRANK=16
set PH_ABS_W_MODE=norm
set RESUME=outputs_real\fix_r4\cond3_ckpt.pt
set RESUME_R_OLD=

echo ============================================================
echo   loss_sep sign fix smoke test  STEPS=%STEPS%
echo   Base: fix_r4 (wcos +0.5077), same start point as mx_A
echo   Expect: gen no longer ~-9.77, no NaN, |d-theta| not pinned
echo ============================================================
echo.

%PY% -u scripts\overfit_cond3.py %STEPS% > outputs_real\fix_sep_50.log 2>&1

echo.
echo ============================================================
echo   Done. Log: outputs_real\fix_sep_50.log
echo ============================================================
