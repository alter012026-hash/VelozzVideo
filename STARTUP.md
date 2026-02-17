# Iniciar o sistema (Windows / PowerShell)

## 1) Preparar ambiente (apenas uma vez)
```
cd D:\Projeto_piloto
npm install
python -m venv video_factory\.venv
video_factory\.venv\Scripts\activate
pip install -r video_factory\requirements.txt
```

## 1.1 Se o venv não existir (atalho rápido)
```
cd D:\Projeto_piloto\video_factory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
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

## 3) Atalho em um comando
- Use `start_all.bat` na raiz. Ele:
  - escolhe porta livre 8000-8010 para a API
  - sobe Ollama
  - sobe API (uvicorn) e espera o ping responder
  - inicia Vite com API_HOST apontando para a porta escolhida

## Observações
- Proxy do front: `/ollama` -> `http://127.0.0.1:11434`, `/api` -> `http://127.0.0.1:%API_PORT%`.
- Saídas de vídeo/áudio ficam em `video_factory/assets/`.
- Se o Vite reclamar de API indisponível, verifique se o uvicorn está rodando ou use `start_all.bat`.
