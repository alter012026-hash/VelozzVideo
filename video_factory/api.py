from __future__ import annotations

import sys
import asyncio
import uuid
import logging
import traceback
from typing import Any, Dict
from pydantic import BaseModel
from fastapi import FastAPI, Query, Response, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import edge_tts
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.tts_generator import synthesize_to_bytes, synthesize_to_bytes_with_metadata  # noqa: E402
from video_factory.config import EDGE_VOICE  # noqa: E402
from video_factory.render_pipeline import RenderRequest, render_script  # noqa: E402
from video_factory.cleanup import purge_old_assets  # noqa: E402
from video_factory.video_renderer import build_effects_preview  # noqa: E402
from video_factory import config  # noqa: E402

# cache de vozes disponíveis (pt-*)
VOICE_BY_ID: dict[str, dict[str, Any]] = {}
VOICE_IDS: set[str] = set()

app = FastAPI(title="VelozzVideo API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve assets (videos/images/audio) for UI playback
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")

# Store of render task statuses (in-memory)
STATUS: Dict[str, Dict[str, Any]] = {}
logger = logging.getLogger("video_factory.api")
API_REVISION = "2026-02-17-audiofix-1"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Retorna detalhes claros quando o corpo recebido nÃ£o bate com RenderRequest.
    Ajuda a depurar erros 422 no frontend.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.on_event("startup")
async def load_voices():
    global VOICE_BY_ID, VOICE_IDS
    try:
        voices = await edge_tts.list_voices()
        pt = [v for v in voices if str(v.get("Locale", "")).startswith("pt-") and v.get("ShortName")]
        VOICE_BY_ID = {}
        for v in pt:
            vid = v["ShortName"]
            locale = str(v.get("Locale") or "")
            # ShortName format example: "pt-BR-AntonioNeural" -> label "AntonioNeural"
            label = vid.split("-", 2)[-1]
            VOICE_BY_ID[vid] = {
                "id": vid,
                "label": label,
                "gender": v.get("Gender", "Unknown"),
                "locale": locale,
                "name": v.get("Name"),
                "friendlyName": v.get("FriendlyName"),
            }
        VOICE_IDS = set(VOICE_BY_ID.keys())
    except Exception:
        VOICE_BY_ID = {}
        VOICE_IDS = set()


@app.get("/api/tts/voices")
async def list_voices():
    if not VOICE_IDS:
        await load_voices()
    return {"voices": list(VOICE_BY_ID.values())}


async def _try_voices(text: str, voices: list[str]) -> tuple[bytes, str]:
    last_exc: Exception | None = None
    for vid in voices:
        try:
            audio_bytes = await synthesize_to_bytes(text, vid)
            return audio_bytes, vid
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc or RuntimeError("TTS failed")


@app.get("/api/tts/preview")
async def tts_preview(text: str = Query(..., min_length=1, max_length=300), voice: str | None = None):
    """
    Prévia TTS usando ordem de providers definida em TTS_PROVIDER (edge,offline,...).
    Não depende de listar vozes do Edge, funciona offline.
    """
    try:
        audio_bytes, meta, ext = await synthesize_to_bytes_with_metadata(text, voice or EDGE_VOICE)
        media = "audio/mpeg" if (ext or "").lower() in {"mp3", "mpeg"} else "audio/wav"
        return Response(content=audio_bytes, media_type=media, headers={"X-Voice-Used": voice or EDGE_VOICE})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}")

@app.post("/api/render")
async def render_video(request: RenderRequest):
    try:
        path = await render_script(request)
        return {"output": str(path), "web_url": _path_to_web(path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}")


class PreviewRequest(BaseModel):
    format: str = "16:9"
    transition: str | None = None
    filter: str | None = None
    colorStrength: float | None = None
    animationType: str | None = None
    captionText: str | None = None
    duration: float | None = 3.5
    ffmpegFilters: str | None = None
    stabilize: bool = False
    aiEnhance: bool = False
    engine: str | None = None


@app.post("/api/render/preview")
async def render_preview(body: PreviewRequest):
    """
    Gera um MP4 curto (3-5s) para pré-visualizar efeitos sem rodar pipeline completo.
    """
    try:
        path = build_effects_preview(
            format_ratio=body.format or "16:9",
            transition=body.transition,
            color_filter=body.filter,
            color_strength=body.colorStrength,
            animation_type=body.animationType,
            caption_text=body.captionText or "Prévia de efeitos",
            duration=body.duration or 3.5,
            ffmpeg_filters=body.ffmpegFilters,
        )
        return {"output": str(path), "web_url": _path_to_web(path)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}")


# ---------- async render with status tracking ----------


def _set_status(task_id: str, **kwargs: Any) -> None:
    STATUS.setdefault(task_id, {})
    STATUS[task_id].update(kwargs)
    STATUS[task_id]["updated_at"] = asyncio.get_event_loop().time()


def _write_task_error_log(task_id: str, tb: str, err: str) -> Path:
    logs_dir = config.ASSETS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"render_{task_id}.log"
    path.write_text(f"task_id={task_id}\nerror={err}\n\n{tb}", encoding="utf-8")
    return path

@app.get("/api/ping")
async def ping():
    return {"status": "ok", "revision": API_REVISION}


@app.post("/api/cleanup")
async def cleanup_cache(max_age_days: int | None = None):
    """
    Limpa arquivos antigos de assets (imagens/áudios/vídeos/temp) baseado em max_age_days.
    Default: config.CACHE_MAX_AGE_DAYS.
    """
    result = purge_old_assets(max_age_days)
    return result

def _path_to_web(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        rel = p.relative_to(ROOT)
        return "/assets/" + rel.as_posix().split("assets/", 1)[-1] if "assets/" in rel.as_posix() else "/assets/" + rel.as_posix()
    except Exception:
        return None


async def _run_render_task(task_id: str, request: RenderRequest) -> None:
    try:
        _set_status(task_id, stage="queued", progress=0.0, message="Iniciando", done=False, error=None, output=None)

        def _progress(stage: str, pct: float, msg: str):
            _set_status(task_id, stage=stage, progress=pct, message=msg)

        path = await render_script(request, progress_cb=_progress)
        web = _path_to_web(path)
        _set_status(task_id, stage="done", progress=1.0, message="Concluído", done=True, output=str(path), web_url=web)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("Render task %s falhou", task_id)
        log_path = _write_task_error_log(task_id, tb, str(exc))
        _set_status(
            task_id,
            stage="error",
            error=str(exc),
            error_trace=tb[-8000:],
            error_log=str(log_path),
            error_log_url=_path_to_web(log_path),
            done=True,
        )


@app.post("/api/render/start")
async def render_start(request: RenderRequest):
    task_id = uuid.uuid4().hex
    asyncio.create_task(_run_render_task(task_id, request))
    return {"task_id": task_id}


@app.get("/api/render/status/{task_id}")
async def render_status(task_id: str):
    data = STATUS.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    return data
