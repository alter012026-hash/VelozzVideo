$ErrorActionPreference = "Stop"

Write-Host "== Verificando Ollama =="
try {
  $resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
  Write-Host "Ollama OK"
} catch {
  Write-Warning "Ollama indisponível em http://127.0.0.1:11434"
}

Write-Host "`n== Verificando FFmpeg no PATH =="
try {
  ffmpeg -version | Select-Object -First 1
  Write-Host "FFmpeg OK"
} catch {
  Write-Warning "FFmpeg não encontrado no PATH"
}
