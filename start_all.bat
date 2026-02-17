@echo off
setlocal
REM === Paths ===
set ROOT=%~dp0
set VENV=%ROOT%video_factory\.venv\Scripts

REM --- Seleciona porta livre para API (tenta 8000-8010) ---
set API_PORT=8000
for /l %%P in (8000,1,8010) do (
  netstat -ano | findstr :%%P >nul
  if errorlevel 1 (
    set API_PORT=%%P
    goto found_port
  )
)
:found_port
echo [INFO] Porta escolhida para API: %API_PORT%
set API_HOST=http://127.0.0.1:%API_PORT%

echo [1/4] Iniciando Ollama (se instalado)...
start "ollama-serve" cmd /c "ollama serve"
timeout /t 2 >nul

echo [2/4] Subindo API FastAPI/uvicorn na porta %API_PORT%...
start "velozz-api" cmd /c "cd /d %ROOT% && set TTS_PROVIDER=edge,offline && %VENV%\python -m uvicorn video_factory.api:app --host 0.0.0.0 --port %API_PORT%"

echo [3/4] Subindo frontend Vite (npm run dev)...
start "velozz-front" cmd /c "cd /d %ROOT% && set API_HOST=%API_HOST% && npm run dev"

echo [4/4] Pronto. Para rodar pipeline CLI manual (opcional):
echo     cd /d %ROOT%video_factory ^&^& %VENV%\python pipeline.py --topic \"Demo\" --aspect 16:9

echo ---
echo Servicos iniciados. Janelas separadas:
echo - ollama-serve
echo - velozz-api  (porta %API_PORT%)
echo - velozz-front (porta 3000)
echo ---
echo Se algum comando falhar, verifique se:
echo - Ollama esta instalado e no PATH
echo - VENV existe em video_factory\.venv
echo - Dependencias npm/pip estao instaladas
echo ---
pause
endlocal
