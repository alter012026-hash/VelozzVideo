from __future__ import annotations

import asyncio
import base64
import logging
import uuid
import re
import shutil
import subprocess
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Callable
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from video_factory import config
from video_factory.tts_generator import synthesize_to_bytes_with_metadata
from video_factory.tts_metadata import WordTiming, word_timings_from_chunks
from video_factory.video_renderer import VideoBuilder
from video_factory.video_renderer import run_ffmpeg_filtergraph

logger = logging.getLogger("render_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _safe_slug(value: str, fallback: str = "render") -> str:
    clean = re.sub(r"[\\/:*?\"<>|]", "", value or "")
    clean = re.sub(r"\s+", "_", clean).strip("_")
    clean = re.sub(r"[^A-Za-z0-9._-]", "", clean)
    return clean or fallback


class RenderScene(BaseModel):
    id: str
    text: str
    visualPrompt: Optional[str] = None
    localImage: Optional[str] = None
    narrationVolume: Optional[float] = 1.0
    trimStartMs: Optional[int] = 0
    trimEndMs: Optional[int] = 0
    audioOffsetMs: Optional[int] = 0
    localSfx: Optional[str] = None
    sfxVolume: Optional[float] = 0.35
    animationType: Optional[str] = None
    transition: Optional[str] = None
    filter: Optional[str] = None


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
    imageScale: float = 1.0
    musicVolume: float = config.MUSIC_VOLUME
    narrationVolume: float = config.NARRATION_VOLUME
    captionFontScale: float = config.CAPTION_FONT_SCALE
    captionBgOpacity: float = config.CAPTION_BG_OPACITY
    captionColor: str = config.CAPTION_COLOR
    captionHighlightColor: str = config.CAPTION_HIGHLIGHT_COLOR
    captionYPct: float = config.CAPTION_Y_PCT
    scriptTitle: Optional[str] = None
    ffmpegFilters: Optional[str] = None  # filtergraph opcional aplicado pós-render
    stabilize: bool = False  # tentativa de estabilização (se libs disponíveis)
    aiEnhance: bool = False  # placeholder p/ futura upscale/denoise
    engine: Optional[str] = None  # moviepy | movielite (quando disponível)
    postOnly: bool = False
    baseVideoUrl: Optional[str] = None


@dataclass
class SceneAsset:
    text: str
    image_path: Path
    audio_path: Path
    narration_volume: float
    trim_start_ms: int
    trim_end_ms: int
    audio_offset_ms: int
    sfx_path: Optional[Path]
    sfx_volume: float
    animation: Optional[str]
    transition: Optional[str]
    color_filter: Optional[str]
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
    ext = (ext or "png").lower()
    ext_alias = {
        "mpeg": "mp3",
        "mpga": "mp3",
        "x-wav": "wav",
        "wave": "wav",
        "x-m4a": "m4a",
        "mp4": "m4a",
        "x-flac": "flac",
        "x-aiff": "aiff",
    }
    ext = ext_alias.get(ext, ext)
    return ext or "png", base64.b64decode(payload)


def _write_asset(prefix: str, data: bytes, extension: str) -> Path:
    folder = config.ASSETS_DIR / prefix
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{uuid.uuid4().hex}.{extension}"
    path.write_bytes(data)
    return path


def _tts_cache_paths(text: str, voice: str, ext: str | None) -> tuple[Path, Path]:
    tts_dir = config.CACHE_DIR / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{voice}\n{text}".encode("utf-8"), usedforsecurity=False).hexdigest()
    audio_path = tts_dir / f"{h}.{ext or 'mp3'}"
    meta_path = tts_dir / f"{h}.json"
    return audio_path, meta_path


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
        path = _write_asset("audio", payload, ext)
        return _coerce_music_for_moviepy(path)
    path = Path(value)
    if path.exists():
        return _coerce_music_for_moviepy(path)
    return None


def _prepare_scene_sfx(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    if value.startswith("data:"):
        ext, payload = _decode_data_url(value)
        path = _write_asset("audio", payload, ext)
        return _coerce_music_for_moviepy(path)
    path = Path(value)
    if path.exists():
        return _coerce_music_for_moviepy(path)
    return None


def _coerce_music_for_moviepy(path: Path) -> Path:
    """
    Normaliza arquivo de música para formatos que o MoviePy costuma ler bem.
    Se extensão for incomum e ffmpeg existir, transcodifica para WAV.
    """
    safe_exts = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}
    suffix = path.suffix.lower()
    if suffix in safe_exts:
        return path
    if not shutil.which("ffmpeg"):
        return path

    target = path.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return target if target.exists() else path
    except Exception:
        return path


def _sanitize_title(value: str) -> str:
    """Remove caracteres problemáticos para nome de arquivo/ffmpeg."""
    if not value:
        return "render"
    safe = re.sub(r'[^A-Za-z0-9._-]+', "_", value.strip())
    # evita nomes vazios ou só underscores
    safe = safe.strip("_.") or "render"
    # limita tamanho para evitar caminhos longos
    return safe[:80]


def _resolve_base_video_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    path_value = raw
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        path_value = parsed.path or ""

    normalized = path_value.replace("\\", "/")
    lower = normalized.lower()
    if lower.startswith("/assets/"):
        candidate = config.ASSETS_DIR / normalized[len("/assets/") :]
    elif "/assets/" in lower:
        candidate = config.ASSETS_DIR / normalized.split("/assets/", 1)[-1]
    elif "assets/" in lower:
        candidate = config.ASSETS_DIR / normalized.split("assets/", 1)[-1]
    else:
        candidate = Path(path_value)

    try:
        candidate = candidate.resolve(strict=False)
    except Exception:
        pass
    return candidate if candidate.exists() else None


async def render_script(
    request: RenderRequest,
    progress_cb: Optional[Callable[[str, float, str, Optional[dict]], None]] = None,
) -> Path:
    if not request.scenes:
        raise RuntimeError("Nenhuma cena informada.")

    voice = request.voice or config.EDGE_VOICE
    scene_assets: List[SceneAsset] = []
    total_scenes = len(request.scenes)

    def _progress(stage: str, pct: float, msg: str = "", detail: Optional[dict] = None) -> None:
        if progress_cb:
            try:
                progress_cb(stage, max(0.0, min(1.0, float(pct))), msg, detail)
            except Exception:
                pass

    if bool(request.postOnly):
        base_video = _resolve_base_video_path(request.baseVideoUrl)
        if not base_video:
            raise RuntimeError("Modo POS exige um video base valido.")

        output = config.ASSETS_DIR / "video" / f"{_sanitize_title(request.scriptTitle or base_video.stem)}_post_{uuid.uuid4().hex}.mp4"
        _progress(
            "post",
            0.0,
            "Reaproveitando video base (sem remontar cenas).",
            {"post_only": True, "base_video": str(base_video), "output_path": str(output)},
        )

        def _build_post_only() -> Path:
            current = base_video
            if request.ffmpegFilters:
                filtered = run_ffmpeg_filtergraph(base_video, request.ffmpegFilters)
                if filtered and filtered.exists():
                    current = filtered
            output.parent.mkdir(parents=True, exist_ok=True)
            if current.resolve(strict=False) != output.resolve(strict=False):
                shutil.copy2(current, output)
            return output

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _build_post_only)
        _progress("post", 1.0, "Pos-processamento concluido", {"post_only": True, "output_path": str(result)})
        logger.info("[POST] Render incremental concluido: %s", result)
        return result

    logger.info("[1/5] Gerando narracoes (%d cenas) - voz=%s", len(request.scenes), voice)
    _progress("tts", 0.0, "Gerando narrações")

    sem = asyncio.Semaphore(getattr(config, "SCENE_PREP_CONCURRENCY", 2))
    scene_assets: list[Optional[SceneAsset]] = [None] * total_scenes

    async def prepare_scene(idx: int, scene: RenderScene) -> None:
        async with sem:
            logger.info("  - Cena %d: texto='%s'", idx + 1, scene.text[:80].strip())
            image_path = _ensure_visual_path(scene, idx)

            # TTS cache
            use_cache = bool(getattr(config, "TTS_CACHE_ENABLED", True))
            audio_bytes: bytes
            metadata: list[dict]
            audio_ext: str | None
            cache_audio: Path | None = None
            cache_meta: Path | None = None
            if use_cache:
                cache_audio, cache_meta = _tts_cache_paths(scene.text, voice, "mp3")
                if cache_audio.exists() and cache_meta.exists():
                    try:
                        audio_bytes = cache_audio.read_bytes()
                        metadata = json.loads(cache_meta.read_text(encoding="utf-8"))
                        audio_ext = cache_audio.suffix.lstrip(".") or "mp3"
                    except Exception:
                        audio_bytes = b""
                        metadata = []
                        audio_ext = None
                else:
                    audio_bytes = b""
                    metadata = []
                    audio_ext = None
            else:
                audio_bytes = b""
                metadata = []
                audio_ext = None

            if not audio_bytes or not metadata:
                audio_bytes, metadata, audio_ext = await synthesize_to_bytes_with_metadata(scene.text, voice)
                if use_cache and cache_audio and cache_meta:
                    try:
                        cache_audio.write_bytes(audio_bytes)
                        cache_meta.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass

            audio_path = _write_asset("audio", audio_bytes, audio_ext or "mp3")

            try:
                scene_narration_volume = max(0.0, min(2.0, float(scene.narrationVolume if scene.narrationVolume is not None else 1.0)))
            except Exception:
                scene_narration_volume = 1.0
            try:
                trim_start_ms = max(0, min(5000, int(scene.trimStartMs if scene.trimStartMs is not None else 0)))
            except Exception:
                trim_start_ms = 0
            try:
                trim_end_ms = max(0, min(5000, int(scene.trimEndMs if scene.trimEndMs is not None else 0)))
            except Exception:
                trim_end_ms = 0
            try:
                audio_offset_ms = max(-3000, min(3000, int(scene.audioOffsetMs if scene.audioOffsetMs is not None else 0)))
            except Exception:
                audio_offset_ms = 0
            sfx_path = _prepare_scene_sfx(scene.localSfx)
            try:
                sfx_volume = max(0.0, min(2.0, float(scene.sfxVolume if scene.sfxVolume is not None else 0.35)))
            except Exception:
                sfx_volume = 0.35
            timings = word_timings_from_chunks(metadata)
            scene_assets[idx] = SceneAsset(
                text=scene.text,
                image_path=image_path,
                audio_path=audio_path,
                narration_volume=scene_narration_volume,
                trim_start_ms=trim_start_ms,
                trim_end_ms=trim_end_ms,
                audio_offset_ms=audio_offset_ms,
                sfx_path=sfx_path,
                sfx_volume=sfx_volume,
                animation=scene.animationType,
                transition=scene.transition,
                color_filter=scene.filter,
                word_timings=timings,
            )
            _progress("tts", (idx + 1) / total_scenes, f"Narração cena {idx+1}/{total_scenes}")

    await asyncio.gather(*(prepare_scene(idx, scene) for idx, scene in enumerate(request.scenes)))
    scene_assets = [asset for asset in scene_assets if asset is not None]
    logger.info("[1/5] Narracoes concluidas")

    logger.info("[2/5] Preparando builder de video (format=%s, transitions=%s)", request.format, request.transitionStyle or "default")
    builder = VideoBuilder(
        format_ratio=request.format,
        transition_style=request.transitionStyle,
        transition_types=request.transitionTypes,
        transition_duration=request.transitionDuration,
        color_filter=request.colorFilter,
        color_strength=request.colorStrength,
        image_scale=request.imageScale,
        narration_volume=request.narrationVolume,
        music_volume=request.musicVolume,
        caption_font_scale=request.captionFontScale,
        caption_bg_opacity=request.captionBgOpacity,
        caption_color=request.captionColor,
        caption_highlight=request.captionHighlightColor,
        caption_y_pct=request.captionYPct,
    )

    logger.info("[3/5] Resolvendo musica de fundo")
    background_music = _prepare_background_music(request.backgroundMusic)
    if background_music:
        logger.info("    Musica encontrada: %s (vol=%.2f)", background_music, request.musicVolume)
    else:
        logger.info("    Nenhuma musica de fundo sera usada")

    output = config.ASSETS_DIR / "video" / f"{_sanitize_title(request.scriptTitle or 'render')}_{uuid.uuid4().hex}.mp4"
    logger.info("[4/5] Montando e concatenando cenas -> %s", output)
    _progress("render", 0.0, "Montando cenas", {"output_path": str(output)})

    def _build() -> Path:
        return builder.build_video(
            visual_paths=[asset.image_path for asset in scene_assets],
            audio_paths=[asset.audio_path for asset in scene_assets],
            output_path=output,
            background_music_path=background_music,
            music_volume=request.musicVolume,
            scene_effects=[asset.animation for asset in scene_assets],
            scene_word_timings=[asset.word_timings for asset in scene_assets],
            scene_texts=[asset.text for asset in scene_assets],
            scene_narration_volumes=[asset.narration_volume for asset in scene_assets],
            scene_trim_start_ms=[asset.trim_start_ms for asset in scene_assets],
            scene_trim_end_ms=[asset.trim_end_ms for asset in scene_assets],
            scene_audio_offset_ms=[asset.audio_offset_ms for asset in scene_assets],
            scene_transitions=[asset.transition for asset in scene_assets],
            scene_filters=[asset.color_filter for asset in scene_assets],
            scene_sfx_paths=[asset.sfx_path for asset in scene_assets],
            scene_sfx_volumes=[asset.sfx_volume for asset in scene_assets],
            progress_cb=lambda pct, msg, detail=None: _progress("render", pct, msg, detail),
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _build)

    # Pós-processamento opcional com FFmpeg filtergraph (não bloqueia render principal)
    _progress("post", 0.2, "Finalizando arquivo")
    if request.ffmpegFilters:
        try:
            _progress("post", 0.4, "Aplicando filtros FFmpeg")
            filtered = await loop.run_in_executor(None, run_ffmpeg_filtergraph, result, request.ffmpegFilters)
            if filtered:
                result = filtered
            _progress("post", 0.95, "Filtros aplicados")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FFmpeg filters falharam: %s", exc)
            _progress("post", 0.95, "Filtros ignorados")

    _progress("post", 1.0, "Render finalizado")
    logger.info("[5/5] Render finalizado: %s", result)
    return result
