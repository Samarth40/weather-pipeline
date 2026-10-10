@echo off
title MLOps Weather AI Pipeline
cd /d "%~dp0"
python run.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying with conda environment...
    E:\anaconda\envs\py311\python.exe run.py
)
pause
