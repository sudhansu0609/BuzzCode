@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   BuzzCode - Rebuild and Install
echo ===================================================
echo.

cd /d "%~dp0\.."

echo [1/2] Building and installing buzzcode binary to %%USERPROFILE%%\.cargo\bin...
cargo install --path crates/buzzcode
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Validating installation...
buzzcode engine doctor --static
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARN] Doctor reported warnings or check failed.
)

echo.
echo ===================================================
echo   BuzzCode successfully installed!
echo   Run 'buzzcode' in any terminal to launch.
echo ===================================================
echo.

if "%~1"=="" pause
