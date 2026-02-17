from __future__ import annotations

import sys
import asyncio
import uuid
from typing import Any, Dict
from fastapi import FastAPI, Query, Response, HTTPException
import edge_tts
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.tts_generator import synthesize_to_bytes  # noqa: E402
from video_factory.config import EDGE_VOICE  # noqa: E402
from video_factory.render_pipeline import RenderRequest, render_script  # noqa: E402

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

# Store of render task statuses (in-memory)
STATUS: Dict[str, Dict[str, Any]] = {}


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
    if not VOICE_IDS:
        await load_voices()
    chosen_voice = voice or EDGE_VOICE
    if chosen_voice not in VOICE_IDS:
        chosen_voice = EDGE_VOICE if EDGE_VOICE in VOICE_IDS else (next(iter(VOICE_IDS), EDGE_VOICE))

    chosen_gender = VOICE_BY_ID.get(chosen_voice, {}).get("gender", "Unknown")
    male_ids = [vid for vid, meta in VOICE_BY_ID.items() if meta.get("gender") == "Male"]
    female_ids = [vid for vid, meta in VOICE_BY_ID.items() if meta.get("gender") == "Female"]

    # ordem de fallback: tenta manter o mesmo gênero quando possível
    if chosen_gender == "Male":
        fallback_order = [chosen_voice] + [vid for vid in male_ids if vid != chosen_voice]
        if EDGE_VOICE in VOICE_IDS and EDGE_VOICE not in fallback_order:
            fallback_order.append(EDGE_VOICE)
        fallback_order += [vid for vid in female_ids if vid not in fallback_order]
    else:
        fallback_order = [chosen_voice]
        if EDGE_VOICE in VOICE_IDS and EDGE_VOICE not in fallback_order:
            fallback_order.append(EDGE_VOICE)
        fallback_order += [vid for vid in female_ids if vid not in fallback_order]
        fallback_order += [vid for vid in male_ids if vid not in fallback_order]
    try:
        audio_bytes, used_voice = await _try_voices(text, fallback_order)
        return Response(content=audio_bytes, media_type="audio/mpeg", headers={"X-Voice-Used": used_voice})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}")


@app.post("/api/render")
async def render_video(request: RenderRequest):
    try:
        path = await render_script(request)
        return {"output": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}")


# ---------- async render with status tracking ----------


def _set_status(task_id: str, **kwargs: Any) -> None:
    STATUS.setdefault(task_id, {})
    STATUS[task_id].update(kwargs)
    STATUS[task_id]["updated_at"] = asyncio.get_event_loop().time()


async def _run_render_task(task_id: str, request: RenderRequest) -> None:
    try:
        _set_status(task_id, stage="queued", progress=0.0, message="Iniciando", done=False, error=None, output=None)

        def _progress(stage: str, pct: float, msg: str):
            _set_status(task_id, stage=stage, progress=pct, message=msg)

        path = await render_script(request, progress_cb=_progress)
        _set_status(task_id, stage="done", progress=1.0, message="Concluído", done=True, output=str(path))
    except Exception as exc:  # noqa: BLE001
        _set_status(task_id, stage="error", error=str(exc), done=True)


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
