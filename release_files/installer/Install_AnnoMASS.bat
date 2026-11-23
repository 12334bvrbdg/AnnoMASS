@echo off
REM Windows: one-time setup. Needs Windows 10/11 with WSL2 and an NVIDIA driver.
setlocal
set "HERE=%~dp0"
wsl.exe --status >nul 2>&1
if errorlevel 1 (
  echo WSL is not installed. Run in an administrator PowerShell:  wsl --install -d Ubuntu
  echo Restart Windows, then run this file again.
  pause & exit /b 1
)
wsl.exe -d Ubuntu --cd "%HERE%" -- bash -lc "bash ./install.sh"
if errorlevel 1 ( echo Installation failed. & pause & exit /b 1 )
REM desktop shortcut -> Start_AnnoMASS.bat
powershell -NoProfile -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\AnnoMASS.lnk'); $s.TargetPath='%HERE%Start_AnnoMASS.bat'; $s.WorkingDirectory='%HERE%'; $s.Save()"
echo Done. Use the AnnoMASS icon on your desktop.
pause
