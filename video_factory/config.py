"""
Centraliza configurações de serviços e paths.
Carrega .env (usando python-dotenv) e oferece defaults seguros.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
CACHE_DIR = ASSETS_DIR / "cache"

# Carrega variáveis do .env na raiz do projeto
load_dotenv(dotenv_path=ROOT_DIR.parent / ".env.local")
load_dotenv(dotenv_path=ROOT_DIR.parent / ".env", override=True)

def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

# LLM / texto
OLLAMA_HOST = env("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = env("LLM_MODEL", "llama3")
LLM_TIMEOUT = int(env("LLM_TIMEOUT", "600"))
LLM_API_KEY = env("LLM_API_KEY")  # OpenAI compatible (opcional)
LLM_API_BASE = env("LLM_API_BASE")  # Ex.: https://api.openai.com/v1

# Imagens (não usados sem SD, mantidos para futura expansão)
IMAGE_API_KEY = env("IMAGE_API_KEY")

# Áudio / TTS
EDGE_VOICE = env("EDGE_VOICE", "pt-BR-ThalitaMultilingualNeural")
ELEVEN_API_KEY = env("ELEVEN_API_KEY")
SPEECHIFY_API_KEY = env("SPEECHIFY_API_KEY")

# Vídeo / render
FPS = int(env("VIDEO_FPS", "30"))
CODEC = env("VIDEO_CODEC", "libx264")
AUDIO_CODEC = env("AUDIO_CODEC", "aac")
ASPECT = env("VIDEO_ASPECT", "16:9")  # 16:9 ou 9:16

# Cache
CACHE_MAX_AGE_DAYS = int(env("CACHE_MAX_AGE_DAYS", "7"))

def ensure_dirs() -> None:
    for path in [ASSETS_DIR, CACHE_DIR, ASSETS_DIR / "images", ASSETS_DIR / "audio", ASSETS_DIR / "video"]:
        path.mkdir(parents=True, exist_ok=True)

ensure_dirs()
