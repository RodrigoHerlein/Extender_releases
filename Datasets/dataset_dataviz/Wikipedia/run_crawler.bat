@echo off
cd /d "%~dp0"
pip install wikipedia-api
python wikipedia_crawler.py
pause
