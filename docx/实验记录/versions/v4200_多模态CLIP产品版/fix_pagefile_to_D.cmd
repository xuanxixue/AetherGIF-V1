@echo off
REM ============================================================
REM  Move the Windows pagefile from C: to D:
REM  Frees ~21GB on C: AND lifts the frozen commit-charge limit
REM  that was causing the OpenBLAS allocation failures.
REM
REM  HOW TO RUN:
REM    Right-click this file -> "Run as administrator" -> Yes
REM    Then REBOOT when prompted.
REM  Pure ASCII on purpose (cmd parses .cmd as GBK).
REM ============================================================
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: not running as Administrator.
    echo   Right-click this file and choose "Run as administrator".
    echo.
    pause
    exit /b 1
)

echo.
echo   Applying: pagefile C: = none, D: = 16384-32768 MB ...

reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v PagingFiles /t REG_MULTI_SZ /d "D:\pagefile.sys 16384 32768" /f
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: registry write failed.
    pause
    exit /b 1
)

echo   Done. Current setting:
reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v PagingFiles

echo.
echo ============================================================
echo   REBOOT NOW to apply. After reboot C: will have ~21GB free
echo   and the commit limit will no longer be frozen at 36.7GB.
echo.
echo   After reboot, start training by double-clicking:
echo     run_train.cmd   (in this same folder)
echo ============================================================
echo.
choice /C YN /M "Reboot now? [Y/N]"
if %ERRORLEVEL% EQU 1 (
    shutdown /r /t 30 /c "Applying pagefile move to D: - reboot in 30s, save your work"
    echo   Rebooting in 30 seconds. Save anything open.
) else (
    echo   Skipped. Remember to reboot manually or the change will not apply.
)
pause