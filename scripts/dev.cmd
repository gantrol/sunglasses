@echo off
setlocal DisableDelayedExpansion
set "devAction=%~1"
set "devInteractive=0"
if "%~2"=="" set "devInteractive=1"

pushd "%~dp0.." || goto path_error
rem Execution policy applies only to this child process; no saved policy is changed.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%CD%\dev.ps1" %*
set "devExitCode=%errorlevel%"
popd

if not "%devExitCode%"=="0" (
    echo.
    echo Sunglasses %devAction% failed. Exit code: %devExitCode%.
    goto finished
)
if /i "%devAction%"=="build" goto finished
if /i "%devAction%"=="package" goto finished
exit /b %devExitCode%

:finished
if "%devInteractive%"=="1" pause
exit /b %devExitCode%

:path_error
echo Unable to open the Sunglasses project directory.
if "%devInteractive%"=="1" pause
exit /b 1
