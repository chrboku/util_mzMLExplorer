REM deactivate ouput of commands
@echo off

REM Check if git is available
where git >nul 2>&1
if errorlevel 1 (
    echo Git is not installed or not in the system PATH. Please visit https://git-scm.com/install/windows, download it and then run the installer. After than try running this script again in a new terminal.
    pause
    exit /b 1
) 

echo Pulling latest changes from the repository...
git pull

pause