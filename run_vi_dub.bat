@echo off
rem Chay VieDub (Windows). Lan dau se tu cai Python 3.10 + thu vien, mat vai phut.
rem Khong phu thuoc vao PATH cua terminal: tu them thu muc uv cai bang winget vao PATH.
set "UV_DIR=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"
set "PATH=%UV_DIR%;%PATH%"
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo [LOI] Chua cai uv. Chay lenh nay roi mo lai cua so nay:
  echo     winget install --id=astral-sh.uv -e
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [CANH BAO] Chua thay ffmpeg trong PATH. Cai bang lenh:
  echo     winget install --id=yt-dlp.FFmpeg -e
  echo Roi dong cua so nay, mo lai de PATH moi co hieu luc.
  pause
  exit /b 1
)

uv run --extra webui vi_dub_web.py %*
pause
