@echo off
:: Build script for mzML Explorer (C++) on Windows

setlocal

set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%build

echo === mzML Explorer C++ Build ===

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

cmake "%SCRIPT_DIR%" -DCMAKE_BUILD_TYPE=Release %*
if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

cmake --build . --config Release --parallel
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo === Build Complete ===
echo Binary: %BUILD_DIR%\Release\mzmlexplorer.exe
echo.
echo To run: %BUILD_DIR%\Release\mzmlexplorer.exe
