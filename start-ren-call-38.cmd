@echo off
cd /d "%~dp0"
call "%~dp0start-ren-call.cmd" --model gemini-3.8-live %*
