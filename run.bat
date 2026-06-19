@echo off
title SpamShield AI - Web Application
color 0A

echo.
echo  ============================================================
echo       SpamShield AI - Email Spam Detection System
echo  ============================================================
echo.
echo   Starting web server...
echo   Open your browser at: http://localhost:5000
echo.
echo   Press Ctrl+C to stop the server.
echo.

"C:\Users\ANTARA\anaconda3\python.exe" "%~dp0app.py"

pause
