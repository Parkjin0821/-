@echo off
cd /d "%~dp0"
start "K-Quant Deck Dashboard" cmd /k "python web_dashboard.py"
