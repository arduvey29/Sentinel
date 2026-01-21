@echo off
echo Starting Silence Index API Server...
echo.
cd /d D:\silence_index
call venv\Scripts\activate.bat
python app.py
