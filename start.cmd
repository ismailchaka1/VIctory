@echo off
rem Starts the game with this fork's diag build, its own cache and data, and a log next to the old runs.
setlocal
cd /d "%~dp0"
set "EXE=%~dp0target\diag\victory.exe"
set "GAME=F:\SeX\MY GAME\game_scarlett_master.exe"
if not exist "%EXE%" (
  echo Build it first: cargo build --profile diag -p victory --features diagnostics
  pause
  exit /b 1
)
set "VICTORY_CACHE_DIR=%LOCALAPPDATA%\victory-fork\cache"
set "VICTORY_DATA_DIR=%LOCALAPPDATA%\victory-fork\data"
set "VICTORY_HOST_WAVE=32"
set "VICTORY_HOST_MAX_WAVE=32"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%t"
set "LOG=F:\VIctory\run-fork-%STAMP%.log"
echo Logging to %LOG%
"%EXE%" run --execute "%GAME%" > "%LOG%" 2>&1
echo Game exited with code %ERRORLEVEL%.
pause
