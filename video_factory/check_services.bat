@echo off
setlocal
echo == Verificando Ollama ==
curl -s -X POST http://127.0.0.1:11434/api/tags >nul && echo Ollama OK || echo Ollama indisponivel

echo.
echo == Verificando FFmpeg ==
ffmpeg -version >nul 2>&1 && echo FFmpeg OK || echo FFmpeg nao encontrado no PATH
endlocal
