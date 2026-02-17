from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
import re

from pydantic import BaseModel, Field

from video_factory import config
from video_factory.tts_generator import synthesize_to_bytes_with_metadata
from video_factory.tts_metadata import WordTiming, word_timings_from_chunks
from video_factory.video_renderer import VideoBuilder

logger = logging.getLogger("render_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


class RenderScene(BaseModel):
    id: str
    text: str
    visualPrompt: Optional[str] = None
    localImage: Optional[str] = None
    animationType: Optional[str] = None


class RenderRequest(BaseModel):
    scenes: List[RenderScene]
    format: str = Field(default="16:9")
    voice: Optional[str] = None
    backgroundMusic: Optional[str] = None
    transitionStyle: Optional[str] = None
    transitionTypes: Optional[List[str]] = None
    transitionDuration: Optional[float] = None
    colorFilter: Optional[str] = None
    colorStrength: float = 0.35
    musicVolume: float = 0.25
    scriptTitle: Optional[str] = None


@dataclass
class SceneAsset:
    image_path: Path
    audio_path: Path
    animation: Optional[str]
    word_timings: List[WordTiming]


def _decode_data_url(data: str) -> tuple[str, bytes]:
    if "," not in data:
        raise ValueError("Formato inválido de data URL")
    header, payload = data.split(",", 1)
    try:
        mime = header.split(":", 1)[1].split(";", 1)[0]
    except IndexError:
        mime = "png"
    ext = mime.split("/")[-1] if "/" in mime else "png"
    return ext or "png", base64.b64decode(payload)


def _write_asset(prefix: str, data: bytes, extension: str) -> Path:
    folder = config.ASSETS_DIR / prefix
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{uuid.uuid4().hex}.{extension}"
    path.write_bytes(data)
    return path


def _ensure_visual_path(scene: RenderScene, idx: int) -> Path:
    if scene.localImage:
        if scene.localImage.startswith("data:"):
            ext, payload = _decode_data_url(scene.localImage)
            return _write_asset("images", payload, ext)
        candidate = Path(scene.localImage)
        if candidate.exists():
            return candidate
    # fallback: render placeholder text
    from PIL import Image, ImageDraw, ImageFont

    width, height = config.VIDEO_FORMATS.get(config.ASPECT, config.VIDEO_FORMATS["16:9"]).values()
    width = int(width)
    height = int(height)
    img = Image.new("RGB", (width, height), color=(6, 8, 15))
    draw = ImageDraw.Draw(img)
    text = (scene.visualPrompt or scene.text or "Cena").strip()
    try:
        font = ImageFont.truetype("arialbd.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.multiline_text((40, height // 2), text, font=font, fill=(255, 255, 255))
    target = config.ASSETS_DIR / "images" / f"placeholder_{uuid.uuid4().hex}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target, format="PNG")
    return target


def _prepare_background_music(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    if value.startswith("data:"):
        ext, payload = _decode_data_url(value)
        return _write_asset("audio", payload, ext)
    path = Path(value)
    if path.exists():
        return path
    return None


def _sanitize_title(value: str) -> str:
    """Remove caracteres problemáticos para nome de arquivo/ffmpeg."""
    if not value:
        return "render"
    safe = re.sub(r'[^A-Za-z0-9._-]+', "_", value.strip())
    # evita nomes vazios ou só underscores
    safe = safe.strip("_.") or "render"
    # limita tamanho para evitar caminhos longos
    return safe[:80]


async def render_script(request: RenderRequest, progress_cb: Optional[Callable[[str, float, str], None]] = None) -> Path:
    if not request.scenes:
        raise RuntimeError("Nenhuma cena informada.")

    voice = request.voice or config.EDGE_VOICE
    scene_assets: List[SceneAsset] = []
    total_scenes = len(request.scenes)

    def _progress(stage: str, pct: float, msg: str = "") -> None:
        if progress_cb:
            try:
                progress_cb(stage, max(0.0, min(1.0, float(pct))), msg)
            except Exception:
                pass

    logger.info("[1/5] Gerando narracoes (%d cenas) - voz=%s", len(request.scenes), voice)
    _progress("tts", 0.0, "Gerando narrações")
    for idx, scene in enumerate(request.scenes):
        logger.info("  - Cena %d: texto='%s'", idx + 1, scene.text[:80].strip())
        image_path = _ensure_visual_path(scene, idx)
        audio_bytes, metadata, audio_ext = await synthesize_to_bytes_with_metadata(scene.text, voice)
        audio_path = _write_asset("audio", audio_bytes, audio_ext or "mp3")
        timings = word_timings_from_chunks(metadata)
        scene_assets.append(SceneAsset(image_path=image_path, audio_path=audio_path, animation=scene.animationType, word_timings=timings))
        _progress("tts", (idx + 1) / total_scenes, f"Narração cena {idx+1}/{total_scenes}")
    logger.info("[1/5] Narracoes concluidas")

    logger.info("[2/5] Preparando builder de video (format=%s, transitions=%s)", request.format, request.transitionStyle or "default")
    builder = VideoBuilder(
        format_ratio=request.format,
        transition_style=request.transitionStyle,
        transition_types=request.transitionTypes,
        transition_duration=request.transitionDuration,
        color_filter=request.colorFilter,
        color_strength=request.colorStrength,
    )

    logger.info("[3/5] Resolvendo musica de fundo")
    background_music = _prepare_background_music(request.backgroundMusic)
    if background_music:
        logger.info("    Musica encontrada: %s (vol=%.2f)", background_music, request.musicVolume)
    else:
        logger.info("    Nenhuma musica de fundo sera usada")

    output = config.ASSETS_DIR / "video" / f"{_sanitize_title(request.scriptTitle or 'render')}_{uuid.uuid4().hex}.mp4"
    logger.info("[4/5] Montando e concatenando cenas -> %s", output)
    _progress("render", 0.0, "Montando cenas")

    def _build() -> Path:
        return builder.build_video(
            visual_paths=[asset.image_path for asset in scene_assets],
            audio_paths=[asset.audio_path for asset in scene_assets],
            output_path=output,
            background_music_path=background_music,
            music_volume=request.musicVolume,
            scene_effects=[asset.animation for asset in scene_assets],
            scene_word_timings=[asset.word_timings for asset in scene_assets],
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _build)
    _progress("render", 1.0, "Render finalizado")
    logger.info("[5/5] Render finalizado: %s", result)
    return result
