@echo off
REM Windows launcher (the Desktop shortcut points here).
wsl.exe -d Ubuntu -- bash -lc "$HOME/.local/bin/annomass"
if errorlevel 1 ( echo AnnoMASS did not start. Run Install_AnnoMASS.bat first. & pause )
