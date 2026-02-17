from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from moviepy.video.fx.SlideIn import SlideIn
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.CrossFadeOut import CrossFadeOut
from moviepy.video.fx.Resize import Resize
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.Clip import Clip as MoviepyClip
from moviepy.audio.AudioClip import AudioClip
import shutil

from video_factory.config import (
    ANIMATION_INTENSITY,
    ANIMATION_STYLE,
    ANIMATION_TYPES,
    ASPECT,
    AUDIO_CODEC,
    CODEC,
    FADE_DURATION,
    FPS,
    TRANSITION_DURATION,
    TRANSITION_STYLE,
    TRANSITION_TYPES,
    VIDEO_FORMATS,
    ASSETS_DIR,
    VIDEO_PRESET,
    VIDEO_THREADS,
    KARAOKE_MAX_HIGHLIGHTS,
    KARAOKE_MIN_HOLD,
)
from video_factory.tts_metadata import WordTiming
from video_factory import config

logger = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    for path in (ASSETS_DIR / "images", ASSETS_DIR / "audio", ASSETS_DIR / "video"):
        path.mkdir(parents=True, exist_ok=True)


_ensure_dirs()

if not hasattr(MoviepyClip, "set_duration"):
    def _compat_set_duration(self, duration, change_end=True):
        """Compat shim when moviepy removes ``set_duration``."""
        self.duration = duration
        if change_end:
            if duration is None:
                self.end = None
            elif self.start is not None:
                self.end = self.start + duration
            else:
                self.end = duration
        else:
            if duration is None:
                raise ValueError("Cannot change clip start when new duration is None")
            if self.end is None:
                raise ValueError("Cannot change clip start: end is undefined")
            self.start = self.end - duration

        for child in (getattr(self, "mask", None), getattr(self, "audio", None)):
            if child is None:
                continue
            if hasattr(child, "set_duration"):
                child.set_duration(duration, change_end=change_end)
            elif hasattr(child, "with_duration"):
                child = child.with_duration(duration, change_end=change_end)
        return self

    MoviepyClip.set_duration = _compat_set_duration


class VideoBuilder:
    """
    Motor de montagem de vídeos que combina MoviePy, transições, efeitos e narração sincronizada.
    """

    def __init__(
        self,
        format_ratio: str = "16:9",
        transition_style: Optional[str] = None,
        transition_types: Optional[List[str]] = None,
        transition_duration: Optional[float] = None,
        color_filter: Optional[str] = None,
        color_strength: float = 0.35,
        narration_volume: float = 1.0,
        music_volume: float = 0.25,
        caption_font_scale: float = 1.0,
        caption_bg_opacity: float = 0.55,
        caption_color: str = "#FFFFFF",
        caption_highlight: str = "#ffd166",
        caption_y_pct: float = 0.82,
    ):
        if format_ratio not in VIDEO_FORMATS:
            raise ValueError(f"Formato inválido: {format_ratio}")
        self.format_ratio = format_ratio
        self.width = VIDEO_FORMATS[format_ratio]["width"]
        self.height = VIDEO_FORMATS[format_ratio]["height"]
        self.fps = FPS
        self.transition_style = (transition_style or TRANSITION_STYLE or "mixed").lower()
        self.transition_types = transition_types or TRANSITION_TYPES
        self.transition_duration = (
            float(transition_duration) if transition_duration is not None else float(TRANSITION_DURATION or FADE_DURATION)
        )
        self.color_filter = (color_filter or "").lower()
        self.color_strength = max(0.0, min(1.5, color_strength))
        self.narration_volume = max(0.0, float(narration_volume))
        self.music_volume = max(0.0, float(music_volume))
        self.caption_font_scale = max(0.6, min(1.6, float(caption_font_scale)))
        self.caption_bg_opacity = max(0.0, min(1.0, float(caption_bg_opacity)))
        self.caption_color = caption_color
        self.caption_highlight = caption_highlight
        self.caption_y_pct = max(0.5, min(0.95, float(caption_y_pct)))

    def _resize_to_cover(self, clip: Any) -> Any:
        """
        Redimensiona mantendo proporção: preenche o quadro e corta o excedente
        para evitar distorção.
        """
        target_aspect = self.width / self.height
        clip_aspect = (clip.w / clip.h) if getattr(clip, "h", 0) else target_aspect

        if clip_aspect > target_aspect:
            clip = clip.resized(height=self.height)
            clip = clip.cropped(x_center=clip.w / 2, width=self.width)
        else:
            clip = clip.resized(width=self.width)
            clip = clip.cropped(y_center=clip.h / 2, height=self.height)
        return clip

    def _pick_transition(self, index: int) -> str:
        style = self.transition_style
        if style == "mixed":
            types = self.transition_types or ["fade"]
            return types[index % len(types)]
        return style

    def _transition_duration(self, clip_duration: float) -> float:
        if clip_duration <= 0:
            return 0.0
        base = self.transition_duration or FADE_DURATION
        return max(0.05, min(base, clip_duration * 0.25))

    def _apply_transition_effects(self, clip: Any, index: int, transition_override: Optional[str] = None) -> tuple[Any, float]:
        transition = (transition_override or self._pick_transition(index) or "none").lower()
        duration = self._transition_duration(getattr(clip, "duration", 0) or 0)
        if duration <= 0 or transition == "none":
            return clip, 0.0

        overlap = duration
        video_effects = []

        if transition == "fade":
            video_effects = [FadeIn(duration), FadeOut(duration)]
        elif transition == "crossfade":
            video_effects = [CrossFadeIn(duration), CrossFadeOut(duration)]
        elif transition.startswith("slide_"):
            side = transition.split("_", 1)[1]
            side = {"up": "top", "down": "bottom"}.get(side, side)
            if side not in {"left", "right", "top", "bottom"}:
                video_effects = [FadeIn(duration), FadeOut(duration)]
            else:
                video_effects = [
                    SlideIn(duration, side),
                    FadeIn(min(0.2, duration)),
                    FadeOut(min(0.2, duration)),
                ]
        else:
            video_effects = [FadeIn(duration), FadeOut(duration)]

        clip = clip.with_effects(video_effects)
        if clip.audio:
            audio = clip.audio.fx(AudioFadeIn(duration)).fx(AudioFadeOut(duration))
            clip = clip.with_audio(audio)
        return clip, max(0.0, float(overlap))

    def _apply_color_filter(self, clip: Any, override: Optional[str] = None, strength_override: Optional[float] = None) -> Any:
        mode = (override or self.color_filter or "none").lower()
        strength = max(0.0, float(self.color_strength if strength_override is None else strength_override))
        if mode in ("none", "off") or strength <= 0:
            return clip

        def _cinematic(img):
            arr = img.astype(np.float32)
            arr *= 1.06
            arr[..., 0] *= 1.03
            arr[..., 2] *= 0.97
            return arr

        def _cool(img):
            arr = img.astype(np.float32)
            arr[..., 1] *= 0.98
            arr[..., 2] *= 1.08
            arr *= 1.02
            return arr

        def _warm(img):
            arr = img.astype(np.float32)
            arr[..., 0] *= 1.08
            arr[..., 2] *= 0.95
            return arr

        def _bw(img):
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            return np.stack([gray, gray, gray], axis=-1)

        def _vibrant(img):
            arr = img.astype(np.float32)
            mean = arr.mean(axis=2, keepdims=True)
            arr = mean + (arr - mean) * 1.18
            arr *= 1.04
            return arr

        def _vhs(img):
            arr = img.astype(np.float32)
            noise = np.random.normal(0, 6, arr.shape)
            arr = arr + noise
            arr[..., 0] *= 1.02
            arr[..., 2] *= 0.94
            return arr

        def _matte(img):
            arr = img.astype(np.float32)
            arr = arr ** 0.96
            arr *= 1.01
            return np.clip(arr + 12, 0, 255)

        filters = {
            "cinematic": _cinematic,
            "cool": _cool,
            "teal": _cool,
            "warm": _warm,
            "bw": _bw,
            "b&w": _bw,
            "grayscale": _bw,
            "vibrant": _vibrant,
            "vhs": _vhs,
            "matte": _matte,
        }
        fn = filters.get(mode)
        if not fn:
            return clip

        alpha = min(1.5, strength)

        def blend(img):
            base = img.astype(np.float32)
            fx = fn(base)
            out = base * (1 - alpha) + fx * alpha
            return np.clip(out, 0, 255).astype(np.uint8)

        return clip.transform(lambda gf, t: blend(gf(t)))

    def _assemble_timeline(self, clips: Sequence[CompositeVideoClip], overlaps: Sequence[float]) -> CompositeVideoClip:
        if not clips:
            raise ValueError("Nenhum clip fornecido para montagem.")

        timeline = []
        current_end = 0.0
        for idx, clip in enumerate(clips):
            overlap = overlaps[idx] if idx < len(overlaps) else 0.0
            start = max(0.0, current_end - max(0.0, overlap))
            clip = clip.with_start(start)
            current_end = start + (clip.duration or 0)
            timeline.append(clip)

        base = CompositeVideoClip(timeline, size=(self.width, self.height))
        return base.subclipped(0, current_end)

    def _add_background_music(
        self,
        video: CompositeVideoClip,
        music_path: Path,
        volume: float = 0.25,
        fade_duration: float = 2.0,
    ) -> CompositeVideoClip:
        if not music_path.exists():
            logger.warning("Música de fundo não encontrada: %s", music_path)
            return video

        music = AudioFileClip(str(music_path))
        fade = min(fade_duration, music.duration * 0.3, video.duration * 0.2)
        music = music.fx(AudioFadeIn(fade)).fx(AudioFadeOut(fade)).volumex(volume)
        loops = []
        t = 0.0
        while t < video.duration:
            loops.append(music.with_start(t))
            t += music.duration
        looped = CompositeAudioClip(loops).subclip(0, video.duration)
        final_audio = CompositeAudioClip([video.audio, looped] if video.audio else [looped])
        return video.with_audio(final_audio)

    def build_video(
        self,
        visual_paths: List[Path],
        audio_paths: List[Path],
        output_path: Path,
        background_music_path: Optional[Path] = None,
        music_volume: float | None = None,
        scene_effects: Optional[Sequence[str]] = None,
        scene_word_timings: Optional[Sequence[List[WordTiming]]] = None,
        scene_transitions: Optional[Sequence[str]] = None,
        scene_filters: Optional[Sequence[str]] = None,
    ) -> Path:
        if len(visual_paths) != len(audio_paths):
            raise ValueError("Número de visuais e áudios precisa ser igual.")

        duration = 0
        clips = []
        overlaps: List[float] = []
        for index, (visual_path, audio_path) in enumerate(zip(visual_paths, audio_paths)):
            audio_clip = AudioFileClip(str(audio_path)).volumex(self.narration_volume)
            animation = scene_effects[index] if scene_effects and index < len(scene_effects) else None
            transition = scene_transitions[index] if scene_transitions and index < len(scene_transitions) else None
            color_filter = scene_filters[index] if scene_filters and index < len(scene_filters) else None
            video_clip = self._build_visual_clip(
                visual_path=visual_path,
                duration=audio_clip.duration,
                animation_type=animation,
                color_filter=color_filter,
            )
            video_clip = video_clip.with_audio(audio_clip)
            timings = scene_word_timings[index] if scene_word_timings and index < len(scene_word_timings) else []
            if timings:
                try:
                    video_clip = self._apply_karaoke_overlay(video_clip, timings)
                except Exception:
                    logger.exception("Falha ao aplicar karaoke na cena %s; seguindo sem karaoke.", index + 1)
            video_clip, overlap = self._apply_transition_effects(video_clip, index, transition_override=transition)
            clips.append(video_clip)
            overlaps.append(overlap)
            duration += audio_clip.duration

        final_video = self._assemble_timeline(clips, overlaps)
        if background_music_path:
            mv = self.music_volume if music_volume is None else music_volume
            final_video = self._add_background_music(
                final_video, background_music_path, volume=mv
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec=CODEC,
            audio_codec=AUDIO_CODEC,
            threads=VIDEO_THREADS,
            preset=VIDEO_PRESET,
        )
        final_video.close()
        for clip in clips:
            clip.close()
        return output_path

    def _apply_karaoke_overlay(self, clip: ImageClip, timings: List[WordTiming]) -> CompositeVideoClip:
        """
        Aplica legendas karaokê com wrap seguro e caixa de fundo sem sair da área visível.
        """
        base_text = " ".join([w.text.strip() for w in timings if w.text.strip()])
        if not base_text or clip.duration <= 0:
            return clip

        font_size = max(26, int(self.height * 0.034 * self.caption_font_scale))
        box_width = int(self.width * 0.9)
        y_pos = int(self.height * self.caption_y_pct)

        # base line (wrap automática)
        txt = TextClip(
            base_text.upper(),
            fontsize=font_size,
            color=self.caption_color,
            font="Arial-Bold",
            method="caption",
            size=(box_width, None),
            align="center",
            stroke_color="#000000",
            stroke_width=3,
        ).with_duration(clip.duration)
        txt = txt.on_color(
            size=(txt.w + 24, txt.h + 18),
            color=(0, 0, 0),
            col_opacity=self.caption_bg_opacity,
            pos=("center", "center"),
        ).with_position(("center", y_pos))

        highlight_clips = []
        tokens = [word for word in timings if word.text.strip()]
        if not tokens:
            return clip

        # Limita quantidade de overlays para evitar estouro de memoria em roteiros longos.
        max_highlights = max(12, int(KARAOKE_MAX_HIGHLIGHTS))
        chunk_size = max(1, int(math.ceil(len(tokens) / max_highlights)))

        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]
            content = " ".join(word.text.strip() for word in chunk if word.text.strip())
            if not content:
                continue
            start = max(0.0, min(chunk[0].start, clip.duration))
            chunk_end = chunk[-1].start + max(chunk[-1].duration, float(KARAOKE_MIN_HOLD))
            end = max(start + float(KARAOKE_MIN_HOLD), min(clip.duration, chunk_end))
            if end <= start:
                continue
            try:
                high = TextClip(
                    content.upper(),
                    fontsize=font_size + 4,
                    color=self.caption_highlight,
                    font="Arial-Bold",
                    method="caption",
                    size=(box_width, None),
                    align="center",
                    stroke_color="#111111",
                    stroke_width=3,
                ).with_start(start).with_duration(end - start)
                high = high.on_color(
                    size=(high.w + 24, high.h + 18),
                    color=(0, 0, 0),
                    col_opacity=self.caption_bg_opacity,
                    pos=("center", "center"),
                ).with_position(("center", y_pos))
                highlight_clips.append(high)
            except Exception:
                logger.debug("Falha ao criar highlight karaoke no chunk %s", i // chunk_size)
                continue

        overlay = CompositeVideoClip([txt, *highlight_clips], size=(self.width, self.height)).with_duration(clip.duration)
        combined = CompositeVideoClip([clip, overlay.with_position((0, 0))], size=(self.width, self.height)).with_duration(clip.duration)
        if clip.audio:
            combined = combined.with_audio(clip.audio)
        return combined

    def _make_text_clip(
        self,
        text: str,
        font_size: int,
        color: str,
        stroke_color: str,
        stroke_width: int,
        opacity: float = 1.0,
    ) -> TextClip:
        base_kwargs = dict(
            text=text,
            font_size=font_size,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            method="label",
            text_align="center",
            horizontal_align="center",
            vertical_align="center",
        )
        fonts = ["Arial-Bold", "DejaVuSans-Bold", "Arial"]
        for font_name in fonts:
            try:
                clip = TextClip(font=font_name, **base_kwargs)
                clip = clip.with_opacity(opacity)
                return clip
            except Exception:
                continue
        clip = TextClip(**base_kwargs)
        return clip.with_opacity(opacity)

    def _build_visual_clip(
        self,
        visual_path: Path,
        duration: float,
        animation_type: Optional[str] = None,
        color_filter: Optional[str] = None,
    ) -> ImageClip:
        ext = visual_path.suffix.lower()
        if ext in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
            clip = VideoFileClip(str(visual_path)).without_audio()
            clip_duration = float(getattr(clip, "duration", 0) or 0)
            if clip_duration and clip_duration < duration:
                loops = int(math.ceil(duration / clip_duration))
                clip = concatenate_videoclips([clip] * loops, method="chain")
            clip = clip.subclipped(0, duration)
            clip = self._resize_to_cover(clip)
        else:
            clip = ImageClip(str(visual_path)).with_duration(duration)
            clip = self._resize_to_cover(clip)
        animation = animation_type or self._default_animation_type(ext)
        clip = self._apply_animation(clip, animation)
        return self._apply_color_filter(clip, override=color_filter)

    def _default_animation_type(self, ext: str) -> str:
        return (
            ANIMATION_STYLE
            if ANIMATION_STYLE and ANIMATION_STYLE != "mixed"
            else ANIMATION_TYPES[hash(ext) % len(ANIMATION_TYPES)] if ANIMATION_TYPES else "zoom_in"
        )

    def _apply_animation(self, clip: ImageClip, animation_type: Optional[str]) -> ImageClip:
        animation_type = (animation_type or "").lower()
        if animation_type == "zoom_out":
            return self._apply_zoom(clip, zoom_in=False)
        if animation_type == "zoom_in_fast":
            return self._apply_zoom(clip, zoom_in=True, intensity=ANIMATION_INTENSITY * 1.5)
        if animation_type == "zoom_out_fast":
            return self._apply_zoom(clip, zoom_in=False, intensity=ANIMATION_INTENSITY * 1.5)
        if animation_type == "kenburns":
            return self._apply_kenburns(clip)
        if animation_type == "pan_left":
            return self._apply_pan(clip, direction="left")
        if animation_type == "pan_right":
            return self._apply_pan(clip, direction="right")
        if animation_type == "pan_up":
            return self._apply_pan(clip, direction="up")
        if animation_type == "pan_down":
            return self._apply_pan(clip, direction="down")
        if animation_type == "rotate_left":
            return self._apply_rotate(clip, direction=-1)
        if animation_type == "rotate_right":
            return self._apply_rotate(clip, direction=1)
        if animation_type == "sway":
            return self._apply_sway(clip)
        if animation_type in {"warp", "warp_in", "warp_out"}:
            direction = -1 if animation_type == "warp_out" else 1
            return self._apply_warp(clip, direction=direction)
        if animation_type == "pulse":
            return self._apply_pulse(clip)
        return self._apply_zoom(clip, zoom_in=True)

    def _apply_zoom(self, clip: ImageClip, zoom_in: bool = True, intensity: float = ANIMATION_INTENSITY) -> ImageClip:
        intensity = max(0.02, float(intensity))

        def zoom_effect(gf, t):
            frame = gf(t)
            progress = np.clip(t / clip.duration, 0.0, 1.0) if clip.duration else 0.0
            delta = intensity * (progress if zoom_in else (1.0 - progress))
            factor = 1.0 + delta
            from PIL import Image

            pil_img = Image.fromarray(frame)
            new_size = (int(pil_img.width * factor), int(pil_img.height * factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            cropped = pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0]))
            return np.array(cropped)

        return clip.transform(zoom_effect, keep_duration=True)

    def _apply_pan(self, clip: ImageClip, direction: str = "left") -> ImageClip:
        direction = direction.lower()
        zoom_boost = 1.0 + (ANIMATION_INTENSITY * 0.6)

        def pan_effect(gf, t):
            frame = gf(t)
            progress = np.clip(t / clip.duration, 0.0, 1.0) if clip.duration else 0.0
            h, w = frame.shape[:2]
            scaled_w = int(w * zoom_boost)
            scaled_h = int(h * zoom_boost)
            from PIL import Image

            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((scaled_w, scaled_h), Image.LANCZOS)
            max_x = max(0, scaled_w - w)
            max_y = max(0, scaled_h - h)

            offset_x = int(max_x * progress)
            offset_y = int(max_y * progress)
            if direction == "right":
                offset_x = max_x - offset_x
            if direction == "up":
                offset_y = max_y - offset_y
            if direction == "down":
                offset_y = offset_y

            cropped = pil_img.crop((offset_x, offset_y, offset_x + w, offset_y + h))
            return np.array(cropped)

        return clip.transform(pan_effect, keep_duration=True)

    def _apply_kenburns(self, clip: ImageClip) -> ImageClip:
        def effect(gf, t):
            frame = gf(t)
            progress = np.clip(t / clip.duration if clip.duration else 0.0, 0.0, 1.0)
            # smoothstep para movimento suave
            progress = progress * progress * (3 - 2 * progress)
            scale = 1.0 + ANIMATION_INTENSITY * (0.8 + progress)
            from PIL import Image

            pil_img = Image.fromarray(frame)
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

            max_x = max(0, new_size[0] - frame.shape[1])
            max_y = max(0, new_size[1] - frame.shape[0])
            x = int(max_x * (0.2 + 0.6 * progress))
            y = int(max_y * (0.15 + 0.7 * progress))
            return np.array(pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0])))

        return clip.transform(effect, keep_duration=True)


def run_ffmpeg_filtergraph(input_path: Path, filtergraph: str) -> Path | None:
    """
    Aplica um filtergraph FFmpeg no arquivo gerado e devolve novo caminho.
    Mantém áudio original, reencoda vídeo com codec configurado.
    """
    if not filtergraph or not shutil.which("ffmpeg"):
        return None
    inp = Path(input_path)
    if not inp.exists():
        return None
    out = inp.with_name(inp.stem + "_ffx.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(inp),
        "-vf",
        filtergraph,
        "-c:v",
        CODEC,
        "-preset",
        VIDEO_PRESET,
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except Exception:
        return None


def _make_placeholder_image(text: str, format_ratio: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    size = VIDEO_FORMATS.get(format_ratio, VIDEO_FORMATS["16:9"])
    width, height = int(size["width"]), int(size["height"])
    img = Image.new("RGB", (width, height), color=(12, 16, 24))
    draw = ImageDraw.Draw(img)
    title = text[:120] or "Prévia"
    subtitle = "MoviePy Preview"
    try:
        font_big = ImageFont.truetype("arialbd.ttf", 72)
        font_small = ImageFont.truetype("arialbd.ttf", 38)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    tw, th = draw.textsize(title, font=font_big)
    sw, sh = draw.textsize(subtitle, font=font_small)
    draw.text(((width - tw) / 2, height * 0.38), title, font=font_big, fill=(235, 242, 255))
    draw.text(((width - sw) / 2, height * 0.55), subtitle, font=font_small, fill=(144, 199, 255))
    target = config.ASSETS_DIR / "cache" / f"preview_{uuid.uuid4().hex}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target, format="PNG")
    return target


def _make_silent_audio(duration: float) -> Path:
    duration = max(0.8, float(duration))
    path = config.ASSETS_DIR / "cache" / f"preview_silence_{uuid.uuid4().hex}.wav"
    def frame_func(t):
        return 0.0
    clip = AudioClip(frame_func, duration=duration, fps=44100)
    path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_audiofile(str(path), fps=44100, nbytes=2, codec="pcm_s16le", logger=None)
    clip.close()
    return path


def build_effects_preview(
    format_ratio: str = "16:9",
    transition: Optional[str] = None,
    color_filter: Optional[str] = None,
    color_strength: Optional[float] = None,
    animation_type: Optional[str] = None,
    caption_text: str = "Prévia de efeitos",
    duration: float = 3.5,
    ffmpeg_filters: Optional[str] = None,
) -> Path:
    """
    Renderiza um MP4 curto para testar combinação de transição + filtro + animação.
    Gera 2 cenas com imagens placeholder e áudio silencioso.
    """
    dur = max(1.5, min(8.0, float(duration)))
    builder = VideoBuilder(
        format_ratio=format_ratio,
        transition_style=transition or "mixed",
        transition_types=[transition] if transition else None,
        transition_duration=min(1.2, max(0.2, TRANSITION_DURATION)),
        color_filter=color_filter,
        color_strength=color_strength if color_strength is not None else 0.35,
        narration_volume=1.0,
        music_volume=0.0,
        caption_font_scale=config.CAPTION_FONT_SCALE,
        caption_bg_opacity=config.CAPTION_BG_OPACITY,
        caption_color=config.CAPTION_COLOR,
        caption_highlight=config.CAPTION_HIGHLIGHT_COLOR,
        caption_y_pct=config.CAPTION_Y_PCT,
    )

    img1 = _make_placeholder_image(f"{caption_text} • Cena A", format_ratio)
    img2 = _make_placeholder_image(f"{caption_text} • Cena B", format_ratio)
    aud1 = _make_silent_audio(dur)
    aud2 = _make_silent_audio(dur)

    output = config.ASSETS_DIR / "video" / f"preview_{uuid.uuid4().hex}.mp4"
    builder.build_video(
        visual_paths=[img1, img2],
        audio_paths=[aud1, aud2],
        output_path=output,
        background_music_path=None,
        music_volume=0.0,
        scene_effects=[animation_type or "kenburns", animation_type or "kenburns"],
        scene_word_timings=[[], []],
        scene_transitions=[transition or "crossfade", transition or "crossfade"],
        scene_filters=[color_filter or "cinematic", color_filter or "cinematic"],
    )
    if ffmpeg_filters:
        filtered = run_ffmpeg_filtergraph(output, ffmpeg_filters)
        return filtered or output
    return output

    def _apply_rotate(self, clip: ImageClip, direction: int = 1) -> ImageClip:
        direction = 1 if direction >= 0 else -1
        max_angle = 4.0 * ANIMATION_INTENSITY * 10

        def effect(gf, t):
            frame = gf(t)
            angle = direction * max_angle * math.sin(2 * math.pi * (t / max(clip.duration or 0.001, 0.001)))
            from PIL import Image

            pil_img = Image.fromarray(frame)
            pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            cropped = pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0]))
            return np.array(cropped)

        return clip.transform(effect, keep_duration=True)

    def _apply_sway(self, clip: ImageClip) -> ImageClip:
        def effect(gf, t):
            frame = gf(t)
            h, w = frame.shape[:2]
            shift = int(np.sin(2 * np.pi * (t / max(clip.duration or 0.001, 0.001))) * w * 0.02)
            return np.roll(frame, shift, axis=1)

        return clip.transform(effect, keep_duration=True)

    def _apply_warp(self, clip: ImageClip, direction: int = 1) -> ImageClip:
        direction = 1 if direction >= 0 else -1
        wobble = ANIMATION_INTENSITY * 0.35

        def effect(gf, t):
            frame = gf(t)
            phase = 2 * np.pi * (t / max(clip.duration or 0.001, 0.001))
            scale = 1.0 + wobble * np.sin(phase) * direction
            angle = 3.0 * wobble * 50 * np.sin(phase * 0.5) * direction
            from PIL import Image

            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
            pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            cropped = pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0]))
            return np.array(cropped)

        return clip.transform(effect, keep_duration=True)

    def _apply_pulse(self, clip: ImageClip) -> ImageClip:
        def effect(gf, t):
            frame = gf(t)
            progress = np.sin(2 * np.pi * (t / max(clip.duration or 0.001, 0.001)))
            factor = 1.0 + ANIMATION_INTENSITY * 0.3 * progress
            from PIL import Image

            pil_img = Image.fromarray(frame)
            new_size = (int(pil_img.width * factor), int(pil_img.height * factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            return np.array(pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0])))

        return clip.transform(effect, keep_duration=True)
