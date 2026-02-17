# Iniciar o sistema (Windows / PowerShell)

## 1) Preparar ambiente (apenas uma vez)
```
cd D:\Projeto_piloto
npm install
python -m venv video_factory\.venv
video_factory\.venv\Scripts\activate
pip install -r video_factory\requirements.txt
```

## 2) Subir serviços em terminais separados

### 2.1 Ollama (LLM local)
```
ollama serve
ollama run llama3   # garante que o modelo esteja carregado
```

### 2.2 API de prévia de voz (FastAPI/uvicorn)
```
cd D:\Projeto_piloto
video_factory\.venv\Scripts\activate
# rode sempre a partir da raiz para o pacote ser encontrado
.\video_factory\.venv\Scripts\python -m uvicorn video_factory.api:app --host 0.0.0.0 --port %API_PORT%
```

### 2.3 Frontend Vite
```
cd D:\Projeto_piloto
npm run dev
```

### 2.4 (Opcional) Pipeline CLI rápido
```
cd D:\Projeto_piloto
video_factory\.venv\Scripts\activate
cd video_factory
python pipeline.py --topic "Seu tema" --aspect 16:9
```

## 3) Atalho em um comando (já existe)
- Arquivo: `start_all.bat` (na raiz). Ele abre janelas para:
  - `ollama serve`
  - API FastAPI (porta %API_PORT% (din�mica))
  - Frontend Vite (porta 3000)
  - Pipeline demo (pode fechar se não quiser)

## Observações
- Proxy do front: `/ollama` -> `http://127.0.0.1:11434`, `/api` -> `http://127.0.0.1:%API_PORT%`.
- Ajuste o texto de prévia de voz na sidebar; clique “Ouvir prévia” para testar o TTS.
- Saídas de vídeo/áudio ficam em `video_factory/assets/`.


