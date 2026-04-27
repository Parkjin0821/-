@echo off
cd /d "%~dp0"
python -m pip install flask requests
python web_dashboard.py
