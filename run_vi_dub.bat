@echo off
rem Chay app long tieng Viet, khong phu thuoc vao PATH cua terminal
set "UV_DIR=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"
set "PATH=%UV_DIR%;%PATH%"
cd /d "%~dp0"
uv run vi_dub_web.py %*
pause
