@echo off
REM Double-click this to open Video Watcher as a simple web page.
cd /d "%~dp0"
echo Installing / updating the bits it needs (first time takes a minute)...
python -m pip install -q --upgrade gradio anthropic faster-whisper yt-dlp
echo.
echo Starting Video Watcher. A browser tab will open, and a phone link is
echo printed below as "Running on public URL". Keep this window open while you use it.
echo.
python app.py
pause
