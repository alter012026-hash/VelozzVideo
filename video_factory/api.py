from __future__ import annotations

import sys
from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.tts_generator import synthesize_to_bytes  # noqa: E402
from video_factory.config import EDGE_VOICE  # noqa: E402

EDGE_VOICES = {
    "pt-BR-ThalitaMultilingualNeural",
    "pt-BR-ThalitaNeural",
    "pt-BR-FranciscaNeural",
    "pt-BR-BrendaNeural",
    "pt-BR-GiovannaNeural",
    "pt-BR-ElzaNeural",
    "pt-BR-LeticiaNeural",
    "pt-BR-LeilaNeural",
    "pt-BR-ManuelaNeural",
    "pt-BR-YaraNeural",
    "pt-BR-AntonioNeural",
    "pt-BR-DonatoNeural",
    "pt-BR-FabioNeural",
    "pt-BR-HumbertoNeural",
    "pt-BR-JulioNeural",
    "pt-BR-NicolauNeural",
    "pt-BR-ValerioNeural",
}

app = FastAPI(title="VelozzVideo API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tts/preview")
async def tts_preview(text: str = Query(..., min_length=1, max_length=300), voice: str | None = None):
    chosen_voice = voice or EDGE_VOICE
    if chosen_voice not in EDGE_VOICES:
        chosen_voice = EDGE_VOICE
    try:
        audio_bytes = await synthesize_to_bytes(text, chosen_voice)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}")
