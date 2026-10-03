@echo off
chcp 65001 >nul
rem Запуск call-assistant. Берёт своё окружение venv, если оно есть,
rem иначе — окружение из соседней рабочей папки call-assistant.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist "..\call-assistant\venv\Scripts\python.exe" (
    set "PY=..\call-assistant\venv\Scripts\python.exe"
) else (
    echo Не найдено окружение Python. Создайте его: python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist ".env" (
    echo Нет файла .env. Скопируйте .env.example в .env и заполните.
    pause
    exit /b 1
)

"%PY%" src\main.py
pause
