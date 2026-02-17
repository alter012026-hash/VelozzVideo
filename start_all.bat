@echo off
setlocal
REM === Paths ===
set ROOT=%~dp0
set VENV=%ROOT%video_factory\.venv\Scripts
set API_READY=0

echo [0/4] Limpando listeners antigos nas portas 8000-8010...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -ge 8000 -and $_.LocalPort -le 8010 } | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { try { Stop-Process -Id $_ -Force -ErrorAction Stop; Write-Host ('[INFO] Finalizado PID ' + $_) } catch {} }"

if not exist "%VENV%\python.exe" (
  echo [WARN] Venv nao encontrado. Criando em video_factory\.venv...
  where python >nul 2>&1
  if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    pause
    exit /b 1
  )
  python -m venv "%ROOT%video_factory\.venv"
  if errorlevel 1 (
    echo [ERRO] Falha ao criar venv.
    pause
    exit /b 1
  )
  echo [INFO] Instalando dependencias Python...
  "%VENV%\python.exe" -m pip install -r "%ROOT%video_factory\requirements.txt"
  if errorlevel 1 (
    echo [ERRO] Falha ao instalar requirements.
    pause
    exit /b 1
  )
)

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
start "velozz-api" cmd /c "cd /d %ROOT% && set TTS_PROVIDER=edge,offline && %VENV%\python -m uvicorn video_factory.api:app --host 0.0.0.0 --port %API_PORT% --reload"

echo [2.5/4] Aguardando API responder para evitar erro do Vite...
for /l %%I in (1,1,10) do (
  powershell -Command "try { iwr http://127.0.0.1:%API_PORT%/api/ping -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
  if not errorlevel 1 (
    set API_READY=1
    goto api_up
  )
  timeout /t 1 >nul
)
:api_up
if "%API_READY%"=="0" (
  echo [ERRO] API nao respondeu em http://127.0.0.1:%API_PORT%/api/ping
  echo [ERRO] Verifique a janela "velozz-api" para identificar a causa.
  echo [ERRO] Frontend nao sera iniciado para evitar erro de proxy do Vite.
  pause
  exit /b 1
)

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
