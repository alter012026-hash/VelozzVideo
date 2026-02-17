<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/temp/1

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

### Dicas rápidas
- Use o botão **Resetar** no topo para limpar o fluxo atual.
- Importe ou exporte roteiros em JSON direto pelo painel de status.
- O indicador de progresso mostra em que etapa (ideia → roteiro → assets → render) você está.

## Backend (VelozzVideo)

Pasta: `video_factory/`  
Runtime: Python 3.10+ com virtualenv.

1. `cd video_factory`
2. `python -m venv .venv && .venv/Scripts/activate` (Windows)  
3. `pip install -r requirements.txt`
4. Copie `.env.example` para `.env` e preencha URLs/chaves (Ollama, SD, Pexels/Pixabay, OpenAI compat, TTS).
5. Suba serviços locais:
   - Ollama em `http://127.0.0.1:11434` (modelo `llama3`)
   - Stable Diffusion (AUTOMATIC1111) com `--api` em `http://127.0.0.1:7860`
   - FFmpeg no PATH
6. Valide com `./check_services.ps1` (ou `.bat`).
7. Render automático CLI: `python pipeline.py --topic "Seu tema" --aspect 16:9`

Módulos principais:
- `script_generator.py` — gera roteiros (Ollama → fallback OpenAI compat).
- `tts_generator.py` — narração Edge TTS.
- `pipeline.py` — orquestra: roteiro -> TTS -> montagem MP4 com MoviePy.

### Orquestração front + backend
- Suba o Ollama local e garanta FFmpeg no PATH; ative o venv do backend.
- Render CLI direto: `cd video_factory && .venv/Scripts/python pipeline.py --topic "Tema" --aspect 16:9`
- Em paralelo rode o front: `npm run dev`. Use export/import de roteiros no front para trocar dados com o backend (arquivos JSON/MP4 produzidos em `video_factory/assets/`).
- Para listar modelos Ollama no front, deixe `OLLAMA_HOST` no `.env.local` ou use o host padrão; o proxy `/ollama` do Vite evita bloqueio de CORS.

## Tecnologias utilizadas

**Frontend**
- Vite + React 19
- Tailwind CDN para utilitários de estilo e glassmorphism customizado
- @google/genai (opcional, para Gemini na nuvem)

**Backend local (video_factory/)**
- Python 3.10+ (venv)
- Ollama client (LLM local)
- Edge TTS (síntese de voz)
- MoviePy + Pillow + NumPy (montagem de vídeo com frames de texto)
- python-dotenv (config), requests (HTTP), tqdm (CLI), soundfile / pydub (áudio)
- Opcional: cliente OpenAI-compat (fallback se configurado LLM_API_BASE/LLM_API_KEY)
- FastAPI + uvicorn (preview de voz via /api/tts/preview)

**Infra/serviços externos esperados**
- Ollama rodando em `http://127.0.0.1:11434` com modelo (ex. `llama3`)
- FFmpeg no PATH
- (Opcional) Gemini API key via AI Studio para geração de vídeo/voz em nuvem

**Build/Dev**
- Node.js (npm scripts: dev/build/preview, backend:check, backend:render)
- TypeScript, JSX (tsconfig com moduleResolution bundler)
