"""
Centraliza configurações de serviços e paths.
Carrega .env (usando python-dotenv) e oferece defaults seguros.
"""
from __future__ import annotations

import os
import multiprocessing
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
CACHE_DIR = ASSETS_DIR / "cache"
CAPTIONS_DIR = ASSETS_DIR / "captions"
ROOT_DIR = ROOT_DIR

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

# Roteiro / narração
WORDS_PER_MINUTE = int(env("WORDS_PER_MINUTE", "160"))
SCENE_COUNTS = {1: int(env("SCENES_PER_MIN_1", "6")), 5: int(env("SCENES_PER_MIN_5", "14"))}
SCRIPT_CHUNK_ENABLED = env("SCRIPT_CHUNK_ENABLED", "0").lower() in {"1", "true", "yes"}
SCRIPT_CHUNK_SCENES = int(env("SCRIPT_CHUNK_SCENES", "4"))
SCRIPT_CHUNK_CONTEXT_SCENES = int(env("SCRIPT_CHUNK_CONTEXT_SCENES", "2"))
SCRIPT_CHUNK_ALLOW_SHORTAGE = env("SCRIPT_CHUNK_ALLOW_SHORTAGE", "1").lower() in {"1", "true", "yes"}
SCRIPT_EXPAND_ENABLED = env("SCRIPT_EXPAND_ENABLED", "1").lower() in {"1", "true", "yes"}
SCRIPT_EXPAND_MAX_PASSES = int(env("SCRIPT_EXPAND_MAX_PASSES", "2"))
SCRIPT_EXPAND_SCENES_PER_PASS = int(env("SCRIPT_EXPAND_SCENES_PER_PASS", "4"))
SCRIPT_EXPAND_MIN_ADD_WORDS = int(env("SCRIPT_EXPAND_MIN_ADD_WORDS", "20"))
SCRIPT_EXPAND_MAX_ADD_WORDS = int(env("SCRIPT_EXPAND_MAX_ADD_WORDS", "80"))
NARRATION_POLISH_ENABLED = env("NARRATION_POLISH_ENABLED", "1").lower() not in {"0", "false", "no"}
NARRATION_POLISH_MAX_WORDS = int(env("NARRATION_POLISH_MAX_WORDS", "26"))

# Áudio / TTS
EDGE_VOICE = env("EDGE_VOICE", "pt-BR-ThalitaMultilingualNeural")
ELEVEN_API_KEY = env("ELEVEN_API_KEY")
SPEECHIFY_API_KEY = env("SPEECHIFY_API_KEY")
TTS_PROVIDER = env("TTS_PROVIDER", "edge,offline")  # ordem de tentativa: ex. "edge,offline"
TTS_CACHE_ENABLED = env("TTS_CACHE_ENABLED", "1").lower() in {"1", "true", "yes"}
SCENE_PREP_CONCURRENCY = max(1, int(env("SCENE_PREP_CONCURRENCY", "2")))

# Vídeo / render
FPS = int(env("VIDEO_FPS", "30"))
CODEC = env("VIDEO_CODEC", "auto")
AUDIO_CODEC = env("AUDIO_CODEC", "aac")
ASPECT = env("VIDEO_ASPECT", "16:9")  # 16:9 ou 9:16

# formatos suportados
VIDEO_FORMATS = {
    "16:9": {"width": 1920, "height": 1080},
    "9:16": {"width": 1080, "height": 1920},
}
VIDEO_FPS = FPS
VIDEO_CODEC = CODEC
VIDEO_AUDIO_CODEC = AUDIO_CODEC
VIDEO_PRESET = env("VIDEO_PRESET", "veryfast")
VIDEO_THREADS = int(env("VIDEO_THREADS", str(max(4, multiprocessing.cpu_count() or 4))))
GPU_ENCODER_PREFERENCE = [
    item.strip().lower()
    for item in env("GPU_ENCODER_PREFERENCE", "h264_nvenc,hevc_nvenc,h264_qsv,hevc_qsv,h264_amf,hevc_amf").split(",")
    if item.strip()
]

# Legendas / mixagem
CAPTION_FONT_SCALE = float(env("CAPTION_FONT_SCALE", "1.0"))  # multiplicador sobre tamanho base
CAPTION_BG_OPACITY = float(env("CAPTION_BG_OPACITY", "0.55"))
CAPTION_COLOR = env("CAPTION_COLOR", "#FFFFFF")
CAPTION_HIGHLIGHT_COLOR = env("CAPTION_HIGHLIGHT_COLOR", "#ffd166")
CAPTION_Y_PCT = float(env("CAPTION_Y_PCT", "0.82"))  # posiÃ§Ã£o vertical (0-1)
NARRATION_VOLUME = float(env("NARRATION_VOLUME", "1.0"))
MUSIC_VOLUME = float(env("MUSIC_VOLUME", "0.25"))
KARAOKE_MAX_HIGHLIGHTS = int(env("KARAOKE_MAX_HIGHLIGHTS", "80"))
KARAOKE_MIN_HOLD = float(env("KARAOKE_MIN_HOLD", "0.08"))

ANIMATION_INTENSITY = float(env("ANIMATION_INTENSITY", "0.35"))
FADE_DURATION = float(env("FADE_DURATION", "0.6"))
MAX_UPSCALE = float(env("MAX_UPSCALE", "1.35"))
ANIMATION_TYPES = [
    item.strip()
    for item in env(
        "ANIMATION_TYPES",
        "kenburns,zoom_in,zoom_out,zoom_in_fast,zoom_out_fast,pan_left,pan_right,pan_up,pan_down,rotate_left,rotate_right,sway,pulse,warp_in,warp_out,dolly_left,dolly_right,orbit,handheld,drift_diag",
    ).split(",")
    if item.strip()
]
ANIMATION_STYLE = env("ANIMATION_STYLE", "mixed")
TRANSITION_STYLE = env("TRANSITION_STYLE", "mixed")
TRANSITION_TYPES = [item.strip() for item in env("TRANSITION_TYPES", "fade,crossfade,slide_left,slide_right,slide_up,slide_down,zoom_in,zoom_out,whip_left,whip_right,flash_white,dip_black").split(",") if item.strip()]
TRANSITION_DURATION = float(env("TRANSITION_DURATION", "0.6"))

# Cache
CACHE_MAX_AGE_DAYS = int(env("CACHE_MAX_AGE_DAYS", "7"))


def ensure_dirs() -> None:
    for path in [
        ASSETS_DIR,
        CACHE_DIR,
        ASSETS_DIR / "images",
        ASSETS_DIR / "audio",
        ASSETS_DIR / "video",
        CAPTIONS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()
